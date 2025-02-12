use crate::tts::tokenize::tokenize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::onn::ort_koko::{self};
use crate::utils;
use crate::utils::fileio::load_json_file;

use espeak_rs::text_to_phonemes;

pub struct TTSOpts<'a> {
    pub txt: &'a str,
    pub lan: &'a str,
    pub style_name: &'a str,
    pub save_path: &'a str,
    pub mono: bool,
    pub speed: f32,
    pub stereo_phase_shift: f32,
    pub initial_silence: Option<usize>,
}

#[derive(Clone)]
pub struct TTSKoko {
    #[allow(dead_code)]
    model_path: String,
    model: Arc<ort_koko::OrtKoko>,
    styles: HashMap<String, Vec<[[f32; 256]; 1]>>,
}

// Function to apply phase shift using an all-pass filter.
// This function takes an audio buffer (slice of f32) and a phase shift value (f32) as input.
// It processes the audio using a first-order all-pass filter to introduce a phase shift,
// effectively creating a stereo widening effect when applied to one channel of a stereo audio signal.
// The all-pass filter works by delaying the input signal and feeding it back into the output,
// which alters the phase response without significantly affecting the amplitude response.
//
// Args:
//   audio: A slice of f32 representing the input audio data.
//   phase_shift: A f32 value between -1.0 and 1.0 representing the desired phase shift.
//                This value determines the filter coefficient 'k', which controls the amount of phase shift.
//
// Returns:
//   A new Vec<f32> containing the phase-shifted audio data.  The length of the output vector
//   is the same as the length of the input audio slice.
//
// The all-pass filter is implemented using the following difference equation:
//   y[n] = k * x[n] + y[n-1] - k * x[n-1]
// where:
//   y[n] is the current output sample
//   x[n] is the current input sample
//   y[n-1] is the previous output sample (initialized to 0.0)
//   x[n-1] is the previous input sample (initialized to 0.0)
//   k is the all-pass filter coefficient, equal to the phase_shift parameter.
fn apply_phase_shift(audio: &[f32], phase_shift: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(audio.len());
    let mut y1 = 0.0;
    let mut x1 = 0.0;

    // all-pass filter coefficient
    let k = phase_shift;

    for &x in audio {
        let y = k * x + y1 - k * x1;
        output.push(y);
        x1 = x;
        y1 = y;
    }

    output
}

impl TTSKoko {
    const MODEL_URL: &str =
        "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.onnx";
    const JSON_DATA_F: &str = "data/voices.json";

    pub const SAMPLE_RATE: u32 = 24000;

    pub async fn new(model_path: &str) -> Self {
        let p = Path::new(model_path);
        if !p.exists() {
            utils::fileio::download_file_from_url(TTSKoko::MODEL_URL, model_path)
                .await
                .expect("download model failed.");
        } else {
            eprintln!("load model from: {}", model_path);
        }

        let model = Arc::new(
            ort_koko::OrtKoko::new(model_path.to_string())
                .expect("Failed to create Kokoro TTS model"),
        );

        // TODO: if(not streaming) { model.print_info(); }
        // model.print_info();

        let mut instance = TTSKoko {
            model_path: model_path.to_string(),
            model,
            styles: HashMap::new(),
        };
        instance.load_voices();
        instance
    }

    fn split_text_into_chunks(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut chunks = Vec::new();

        // First split by sentences - using common sentence ending punctuation
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '?' || c == '!' || c == ';')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();

        for sentence in sentences {
            // Clean up the sentence and add back punctuation
            let sentence = format!("{}.", sentence.trim());

            // Convert to phonemes to check token count
            let sentence_phonemes = text_to_phonemes(&sentence, "en", None, true, false)
                .unwrap_or_default()
                .join("");
            let token_count = tokenize(&sentence_phonemes).len();

            if token_count > max_tokens {
                // If single sentence is too long, split by words
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    let test_chunk = if word_chunk.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", word_chunk, word)
                    };

                    let test_phonemes = text_to_phonemes(&test_chunk, "en", None, true, false)
                        .unwrap_or_default()
                        .join("");
                    let test_tokens = tokenize(&test_phonemes).len();

                    if test_tokens > max_tokens {
                        if !word_chunk.is_empty() {
                            chunks.push(word_chunk);
                        }
                        word_chunk = word.to_string();
                    } else {
                        word_chunk = test_chunk;
                    }
                }

                if !word_chunk.is_empty() {
                    chunks.push(word_chunk);
                }
            } else if !current_chunk.is_empty() {
                // Try to append to current chunk
                let test_text = format!("{} {}", current_chunk, sentence);
                let test_phonemes = text_to_phonemes(&test_text, "en", None, true, false)
                    .unwrap_or_default()
                    .join("");
                let test_tokens = tokenize(&test_phonemes).len();

                if test_tokens > max_tokens {
                    // If combining would exceed limit, start new chunk
                    chunks.push(current_chunk);
                    current_chunk = sentence;
                } else {
                    current_chunk = test_text;
                }
            } else {
                current_chunk = sentence;
            }
        }

        // Add the last chunk if not empty
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    pub fn tts_raw_audio(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Split text into appropriate chunks
        let chunks = self.split_text_into_chunks(txt, 500); // Using 500 to leave 12 tokens of margin
        let mut final_audio = Vec::new();

        // Get style vectors once
        let styles = self.mix_styles(style_name)?;

        for chunk in chunks {
            // Convert chunk to phonemes
            let phonemes = text_to_phonemes(&chunk, lan, None, true, false)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?
                .join("");

            let mut tokens = tokenize(&phonemes);
            for _ in 0..initial_silence.unwrap_or(0) {
                tokens.insert(0, 30);
            }
            let tokens = vec![tokens];

            match self.model.infer(tokens, styles.clone(), speed) {
                Ok(chunk_audio) => {
                    let chunk_audio: Vec<f32> = chunk_audio.iter().cloned().collect();
                    final_audio.extend_from_slice(&chunk_audio);
                }
                Err(e) => {
                    eprintln!("Error processing chunk: {:?}", e);
                    eprintln!("Chunk text was: {:?}", chunk);
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Chunk processing failed: {:?}", e),
                    )));
                }
            }
        }

        Ok(final_audio)
    }

    pub fn tts(
        &self,
        TTSOpts {
            txt,
            lan,
            style_name,
            save_path,
            mono,
            speed,
            stereo_phase_shift,
            initial_silence,
        }: TTSOpts,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let audio = self.tts_raw_audio(&txt, lan, style_name, speed, initial_silence)?;

        // Save to file
        let channels = if mono { 1 } else { 2 };
        let spec = hound::WavSpec {
            channels,
            sample_rate: TTSKoko::SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(save_path, spec)?;

        if mono {
            // Mono output
            for &sample in &audio {
                writer.write_sample(sample)?;
            }
        } else if stereo_phase_shift != 0.0 {
            let shifted_audio = apply_phase_shift(&audio, stereo_phase_shift);

            for i in 0..audio.len() {
                writer.write_sample(audio[i])?; // Left channel (original)
                writer.write_sample(shifted_audio[i])?; // Right channel (phase-shifted)
            }
        } else {
            // Stereo from mono (duplicate to both channels)
            for &sample in &audio {
                writer.write_sample(sample)?;
                writer.write_sample(sample)?;
            }
        }

        writer.finalize()?;
        eprintln!("Audio saved to {}", save_path);
        Ok(())
    }

    pub fn mix_styles(
        &self,
        style_name: &str,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !style_name.contains("+") {
            if let Some(style) = self.styles.get(style_name) {
                let styles = vec![style[0][0].to_vec()];
                Ok(styles)
            } else {
                Err(format!("can not found from styles_map: {}", style_name).into())
            }
        } else {
            eprintln!("parsing style mix");
            let styles: Vec<&str> = style_name.split('+').collect();

            let mut style_names = Vec::new();
            let mut style_portions = Vec::new();

            for style in styles {
                if let Some((name, portion)) = style.split_once('.') {
                    if let Ok(portion) = portion.parse::<f32>() {
                        style_names.push(name);
                        style_portions.push(portion * 0.1);
                    }
                }
            }
            eprintln!("styles: {:?}, portions: {:?}", style_names, style_portions);

            let mut blended_style = vec![vec![0.0; 256]; 1];

            for (name, portion) in style_names.iter().zip(style_portions.iter()) {
                if let Some(style) = self.styles.get(*name) {
                    let style_slice = &style[0][0]; // This is a [256] array
                                                    // Blend into the blended_style
                    for j in 0..256 {
                        blended_style[0][j] += style_slice[j] * portion;
                    }
                }
            }
            Ok(blended_style)
        }
    }

    pub fn load_voices(&mut self) {
        // load from json, get styles
        let values = load_json_file(TTSKoko::JSON_DATA_F);
        if let Ok(values) = values {
            if let Some(obj) = values.as_object() {
                for (key, value) in obj {
                    // Check if value is an array
                    if let Some(outer_array) = value.as_array() {
                        // Define target multidimensional vec
                        let mut tensor = vec![[[0.0; 256]; 1]; 511];

                        // Iterate through outer array (511 elements)
                        for (i, inner_value) in outer_array.iter().enumerate() {
                            if let Some(middle_array) = inner_value.as_array() {
                                // Iterate through middle array (1 element)
                                for (j, inner_inner_value) in middle_array.iter().enumerate() {
                                    if let Some(inner_array) = inner_inner_value.as_array() {
                                        // Iterate through inner array (256 elements)
                                        for (k, number) in inner_array.iter().enumerate() {
                                            if let Some(num) = number.as_f64() {
                                                tensor[i][j][k] = num as f32;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Insert multidimensional array into HashMap
                        self.styles.insert(key.clone(), tensor);
                    }
                }
            }

            eprintln!("voice styles loaded: {}", self.styles.len());
            let mut keys: Vec<_> = self.styles.keys().cloned().collect();
            keys.sort();
            eprintln!("{:?}", keys);
            eprintln!(
                "{:?} {:?}",
                self.styles.keys().next(),
                self.styles.keys().nth(1)
            );
        }
    }
}
