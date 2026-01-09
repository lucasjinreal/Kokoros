use clap::{Parser, Subcommand};
use kokoros::{
    tts::koko::{TTSKoko, TTSOpts},
    utils::wav::{WavHeader, write_audio_chunk},
};
use std::net::{IpAddr, SocketAddr};
use std::{
    fs::{self},
    io::Write,
    path::Path,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing_subscriber::fmt::time::FormatTime;

/// Custom Unix timestamp formatter for tracing logs
struct UnixTimestampFormatter;

impl FormatTime for UnixTimestampFormatter {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let timestamp = format!("{}.{:06}", now.as_secs(), now.subsec_micros());
        write!(w, "{}", timestamp)
    }
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Generate speech for a string of text
    #[command(alias = "t", long_flag_alias = "text", short_flag_alias = 't')]
    Text {
        /// Text to generate speech for
        #[arg(
            default_value = "Hello, This is Kokoro, your remarkable AI TTS. It's a TTS model with merely 82 million parameters yet delivers incredible audio quality.
                This is one of the top notch Rust based inference models, and I'm sure you'll love it. If you do, please give us a star. Thank you very much.
                As the night falls, I wish you all a peaceful and restful sleep. May your dreams be filled with joy and happiness. Good night, and sweet dreams!"
        )]
        text: String,

        /// Path to output the WAV file to on the filesystem
        #[arg(
            short = 'o',
            long = "output",
            value_name = "OUTPUT_PATH",
            default_value = "tmp/output.wav"
        )]
        save_path: String,
    },

    /// Read from a file path and generate a speech file for each line
    #[command(alias = "f", long_flag_alias = "file", short_flag_alias = 'f')]
    File {
        /// Filesystem path to read lines from
        input_path: String,

        /// Format for the output path of each WAV file, where {line} will be replaced with the line number
        #[arg(
            short = 'o',
            long = "output",
            value_name = "OUTPUT_PATH_FORMAT",
            default_value = "tmp/output_{line}.wav"
        )]
        save_path_format: String,
    },

    /// Continuously read from stdin to generate speech, outputting to stdout, for each line
    #[command(aliases = ["stdio", "stdin", "-"], long_flag_aliases = ["stdio", "stdin"])]
    Stream,

    /// Start an OpenAI-compatible HTTP server
    #[command(name = "openai", alias = "oai", long_flag_aliases = ["oai", "openai"])]
    OpenAI {
        /// IP address to bind to (typically 127.0.0.1 or 0.0.0.0)
        #[arg(long, default_value_t = [0, 0, 0, 0].into())]
        ip: IpAddr,

        /// Port to expose the HTTP server on
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
}

#[derive(Parser, Debug)]
#[command(name = "kokoros")]
#[command(version = "0.1")]
#[command(author = "Lucas Jin")]
struct Cli {
    /// A language identifier from
    /// https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
    #[arg(
        short = 'l',
        long = "lan",
        value_name = "LANGUAGE",
        default_value = "en-us"
    )]
    lan: String,

    /// Path to the Kokoro v1.0 ONNX model on the filesystem
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL_PATH",
        default_value = "checkpoints/kokoro-v1.0.onnx"
    )]
    model_path: String,

    /// Path to the voices data file on the filesystem
    #[arg(
        short = 'd',
        long = "data",
        value_name = "DATA_PATH",
        default_value = "data/voices-v1.0.bin"
    )]
    data_path: String,

    /// Which single voice to use or voices to combine to serve as the style of speech
    #[arg(
        short = 's',
        long = "style",
        value_name = "STYLE",
        // if users use `af_sarah.4+af_nicole.6` as style name
        // then we blend it, with 0.4*af_sarah + 0.6*af_nicole
        default_value = "af_sarah.4+af_nicole.6"
    )]
    style: String,

    /// Rate of speech, as a coefficient of the default
    /// (i.e. 0.0 to 1.0 is slower than default,
    /// whereas 1.0 and beyond is faster than default)
    #[arg(
        short = 'p',
        long = "speed",
        value_name = "SPEED",
        default_value_t = 1.0
    )]
    speed: f32,

    /// Output audio in mono (as opposed to stereo)
    #[arg(long = "mono", default_value_t = false)]
    mono: bool,

    /// Initial silence duration in tokens
    #[arg(long = "initial-silence", value_name = "INITIAL_SILENCE")]
    initial_silence: Option<usize>,

    /// Also output a sidecar TSV file with word-level timestamps
    #[arg(long = "timestamps", default_value_t = false, global = true)]
    timestamps: bool,

    /// Number of TTS instances for parallel processing
    #[arg(long = "instances", value_name = "INSTANCES", default_value_t = 2)]
    instances: usize,

    #[command(subcommand)]
    mode: Mode,
}

fn derive_tsv_path_from_wav(path: &str) -> String {
    let p = Path::new(path);
    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
        if let Some(parent) = p.parent() {
            return parent
                .join(format!("{stem}.tsv"))
                .to_string_lossy()
                .to_string();
        }
        return format!("{stem}.tsv");
    }
    // Fallback: just append .tsv
    format!("{path}.tsv")
}

fn write_wav_file(
    path: &str,
    samples: &[f32],
    sample_rate: u32,
    mono: bool,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let channels: u16 = if mono { 1 } else { 2 };
    let bits_per_sample: u16 = 32; // f32
    let bytes_per_sample: u32 = (bits_per_sample as u32) / 8;
    let block_align: u16 = channels * bits_per_sample / 8;
    let byte_rate: u32 = sample_rate * (block_align as u32);

    // Data size in bytes
    let num_frames: usize = samples.len();
    let total_samples_to_write: usize = if mono { num_frames } else { num_frames * 2 };
    let data_size: u32 = (total_samples_to_write as u32) * bytes_per_sample;
    let riff_chunk_size: u32 = 36 + data_size; // 4 + (8+16) + (8+data)

    let mut f = File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&riff_chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&(16u32).to_le_bytes())?; // PCM fmt chunk size
    f.write_all(&(3u16).to_le_bytes())?; // IEEE float = 3
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;

    // write samples
    if mono {
        for &s in samples {
            f.write_all(&s.to_le_bytes())?;
        }
    } else {
        for &s in samples {
            f.write_all(&s.to_le_bytes())?; // left
            f.write_all(&s.to_le_bytes())?; // right (duplicate for simple stereo)
        }
    }

    Ok(())
}

fn write_tsv(path: &str, alignments: &[(String, f32, f32)]) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create(path)?;
    f.write_all(b"word\tstart_sec\tend_sec\n")?;
    for (w, s, e) in alignments {
        // Use 3 decimal places by default
        let line = format!("{}\t{:.3}\t{:.3}\n", w, s, e);
        f.write_all(line.as_bytes())?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with Unix timestamp format and environment-based log level
    tracing_subscriber::fmt()
        .with_timer(UnixTimestampFormatter)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let Cli {
            lan,
            model_path,
            data_path,
            style,
            speed,
            initial_silence,
            mono,
            timestamps,
            instances,
            mode,
        } = Cli::parse();

        let tts = TTSKoko::new(&model_path, &data_path).await;

        match mode {
            Mode::File {
                input_path,
                save_path_format,
            } => {
                let file_content = fs::read_to_string(input_path)?;
                for (i, line) in file_content.lines().enumerate() {
                    let stripped_line = line.trim();
                    if stripped_line.is_empty() {
                        continue;
                    }

                    let save_path = save_path_format.replace("{line}", &i.to_string());
                    if timestamps {
                        match tts.tts_timestamped_raw_audio(
                            stripped_line,
                            &lan,
                            &style,
                            speed,
                            initial_silence,
                            None,
                            None,
                            None,
                        ) {
                            Ok(Some((audio, words))) => {
                                // Write WAV
                                // Note: current engine uses 24kHz
                                write_wav_file(&save_path, &audio, 24_000, mono)?;

                                // Write TSV sidecar
                                let tsv_path = derive_tsv_path_from_wav(&save_path);
                                let rows: Vec<(String, f32, f32)> = words
                                    .into_iter()
                                    .map(|w| (w.word, w.start_sec, w.end_sec))
                                    .collect();
                                write_tsv(&tsv_path, &rows)?;
                                eprintln!("Audio saved to {}", save_path);
                                eprintln!("Timestamps saved to {}", tsv_path);
                            }
                            Ok(None) => {
                                eprintln!("No audio produced for line {}", i + 1);
                            }
                            Err(e) => {
                                eprintln!("Error processing line {}: {}", i + 1, e);
                            }
                        }
                    } else {
                        tts.tts(TTSOpts {
                            txt: stripped_line,
                            lan: &lan,
                            style_name: &style,
                            save_path: &save_path,
                            mono,
                            speed,
                            initial_silence,
                        })?;
                    }
                }
            }

            Mode::Text { text, save_path } => {
                let s = std::time::Instant::now();
                if timestamps {
                    match tts.tts_timestamped_raw_audio(
                        &text,
                        &lan,
                        &style,
                        speed,
                        initial_silence,
                        None,
                        None,
                        None,
                    ) {
                        Ok(Some((audio, words))) => {
                            write_wav_file(&save_path, &audio, 24_000, mono)?;
                            let tsv_path = derive_tsv_path_from_wav(&save_path);
                            let rows: Vec<(String, f32, f32)> = words
                                .into_iter()
                                .map(|w| (w.word, w.start_sec, w.end_sec))
                                .collect();
                            write_tsv(&tsv_path, &rows)?;
                            eprintln!("Audio saved to {}", save_path);
                            eprintln!("Timestamps saved to {}", tsv_path);
                        }
                        Ok(None) => {
                            eprintln!("No audio produced for input text");
                        }
                        Err(e) => {
                            eprintln!("Error processing input text: {}", e);
                        }
                    }
                } else {
                    tts.tts(TTSOpts {
                        txt: &text,
                        lan: &lan,
                        style_name: &style,
                        save_path: &save_path,
                        mono,
                        speed,
                        initial_silence,
                    })?;
                }
                println!("Time taken: {:?}", s.elapsed());
                let words_per_second =
                    text.split_whitespace().count() as f32 / s.elapsed().as_secs_f32();
                println!("Words per second: {:.2}", words_per_second);
            }

            Mode::OpenAI { ip, port } => {
                // Create multiple independent TTS instances for parallel processing
                let mut tts_instances = Vec::new();
                for i in 0..instances {
                    tracing::info!(
                        "Initializing TTS instance [{}] ({}/{})",
                        format!("{:02x}", i),
                        i + 1,
                        instances
                    );
                    let instance = TTSKoko::new(&model_path, &data_path).await;
                    tts_instances.push(instance);
                }
                let app = kokoros_openai::create_server(tts_instances).await;
                let addr = SocketAddr::from((ip, port));
                let binding = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("Starting OpenAI-compatible HTTP server on {}", addr);
                kokoros_openai::serve(binding, app.into_make_service()).await?;
            }

            Mode::Stream => {
                let stdin = tokio::io::stdin();
                let reader = BufReader::new(stdin);
                let mut lines = reader.lines();

                // Use std::io::stdout() for sync writing
                let mut stdout = std::io::stdout();

                eprintln!(
                    "Entering streaming mode. Type text and press Enter. Use Ctrl+D to exit."
                );

                // Write WAV header first
                let header = WavHeader::new(1, 24000, 32);
                header.write_header(&mut stdout)?;
                stdout.flush()?;

                while let Some(line) = lines.next_line().await? {
                    let stripped_line = line.trim();
                    if stripped_line.is_empty() {
                        continue;
                    }

                    // Process the line and get audio data
                    match tts.tts_raw_audio(
                        &stripped_line,
                        &lan,
                        &style,
                        speed,
                        initial_silence,
                        None,
                        None,
                        None,
                    ) {
                        Ok(raw_audio) => {
                            // Write the raw audio samples directly
                            write_audio_chunk(&mut stdout, &raw_audio)?;
                            stdout.flush()?;
                            eprintln!("Audio written to stdout. Ready for another line of text.");
                        }
                        Err(e) => eprintln!("Error processing line: {}", e),
                    }
                }
            }
        }

        Ok(())
    })
}
