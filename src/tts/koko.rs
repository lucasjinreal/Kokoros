use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use ndarray::{ArrayBase, ArrayView1, IxDyn, OwnedRepr};

use crate::onn::ort_base::OrtBase;
use crate::onn::ort_koko::{self, OrtKoko};
use crate::utils;
use crate::utils::fileio::load_json_file;

pub struct TTSKoko {
    model_path: String,
    model: ort_koko::OrtKoko,
    styles: HashMap<String, [[[f32; 256]; 1]; 511]>,
}

impl TTSKoko {
    const MODEL_URL: &str =
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx";
    const JSON_DATA_F: &str = "data/voices.json";

    const SAMPLE_RATE: u32 = 24000;

    pub fn new(model_path: &str) -> Self {
        let p = Path::new(model_path);
        if !p.exists() {
            utils::fileio::download_file_from_url(TTSKoko::MODEL_URL, model_path)
                .expect("download model failed.");
        } else {
            println!("load model from: {}", model_path);
        }

        let model = ort_koko::OrtKoko::new(model_path.to_string())
            .expect("Failed to create Kokoro TTS model");

        model.print_info();

        let mut instance = TTSKoko {
            model_path: model_path.to_string(),
            model,
            styles: HashMap::new(),
        };
        instance.load_voices();
        instance
    }

    pub fn tts(&self, style_name: &str) {
        // given string, forward result
        println!("hello, going to tts.");

        // tokens, styles, speed
        // i32, i32

        let tokens = [[
            0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 87, 158, 54, 46, 5, 0,
        ]];
        let tokens: Vec<Vec<i64>> = tokens.iter().map(|&row| row.to_vec()).collect();
        let styles: Vec<Vec<f32>> = self
            .styles
            .values()
            .nth(1)
            .map(|style| {
                // 假设 style 是 [[[f32; 256]; 1]; 511] 格式
                // 我们只取第一个元素，这样就得到 [1, 256] 的形状
                vec![style[0][0].to_vec()]
            })
            .unwrap_or_else(|| vec![vec![0.0; 256]]);

        let out = self.model.infer(tokens, styles);
        println!("output: {:?}", out);

        // save out to audio.wav
        if let Ok(out) = out {
            self.process_and_save_audio(out, 512)
                .expect("save audio failed.");
        }
    }

    fn process_and_save_audio(
        &self,
        output: ArrayBase<OwnedRepr<f32>, IxDyn>,
        phonemes_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start_t = Instant::now();

        // 将输出转换为标准 Vec
        let audio: Vec<f32> = output.iter().cloned().collect();

        // 计算音频时长
        let audio_duration = audio.len() as f32 / TTSKoko::SAMPLE_RATE as f32;

        // 计算创建时间
        let create_duration = start_t.elapsed().as_secs_f32();

        // 计算加速因子
        let speedup_factor = audio_duration / create_duration;

        println!(
            "Created audio in length of {:.2}s for {} phonemes in {:.2}s ({:.2}x real-time)",
            audio_duration, phonemes_len, create_duration, speedup_factor
        );

        // 保存为 WAV 文件
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TTSKoko::SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create("output.wav", spec)?;

        for &sample in &audio {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;

        println!("Audio saved to output.wav");

        Ok(())
    }

    pub fn load_voices(&mut self) {
        // load from json, get styles
        let values = load_json_file(TTSKoko::JSON_DATA_F);
        if let Ok(values) = values {
            if let Some(obj) = values.as_object() {
                for (key, value) in obj {
                    // 检查值是否为数组
                    if let Some(outer_array) = value.as_array() {
                        // 定义目标多维数组
                        let mut array_3d = [[[0.0; 256]; 1]; 511];

                        // 遍历外层数组（511 个元素）
                        for (i, inner_value) in outer_array.iter().enumerate() {
                            if let Some(middle_array) = inner_value.as_array() {
                                // 遍历中层数组（1 个元素）
                                for (j, inner_inner_value) in middle_array.iter().enumerate() {
                                    if let Some(inner_array) = inner_inner_value.as_array() {
                                        // 遍历内层数组（256 个元素）
                                        for (k, number) in inner_array.iter().enumerate() {
                                            if let Some(num) = number.as_f64() {
                                                array_3d[i][j][k] = num as f32;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // 将多维数组插入 HashMap
                        self.styles.insert(key.clone(), array_3d);
                    }
                }
            }

            println!("voice styles loaded: {}", self.styles.len());
            println!("{:?}", self.styles.keys());
            println!(
                "{:?} {:?}",
                self.styles.keys().next(),
                self.styles.keys().nth(1)
            );
        }
    }
}