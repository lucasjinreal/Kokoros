use clap::{Parser, Subcommand, ValueEnum};
use kokoros::{
    tts::koko::{TTSKoko, TTSOpts},
    utils::wav::{write_audio_chunk, WavHeader},
};
use std::net::{IpAddr, SocketAddr};
use std::{
    fs::{self},
    io::Write,
    path::Path,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing_subscriber::fmt::time::FormatTime;

/// Logging destination options
#[derive(Debug, Clone, ValueEnum)]
enum LogDestination {
    /// Log to console only
    Cli,
    /// Log to file only
    File,
    /// Log to both console and file
    All,
    /// Disable all logging
    None,
}

impl Default for LogDestination {
    fn default() -> Self {
        LogDestination::Cli
    }
}

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

    /// Number of TTS instances for parallel processing
    #[arg(long = "instances", value_name = "INSTANCES", default_value_t = 2)]
    instances: usize,

    /// Configure logging output destination
    #[arg(long = "log", value_enum, default_value_t = LogDestination::Cli)]
    log_destination: LogDestination,

    /// Custom log file path (defaults to logs/kokoros-http.log with daily rotation)
    #[arg(long = "log-file", value_name = "LOG_FILE")]
    log_file: Option<String>,

    #[command(subcommand)]
    mode: Mode,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    
    // Initialize logging based on user choice
    match args.log_destination {
        LogDestination::None => {
            // No logging - use a minimal subscriber that filters everything out
            use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
            tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new("off"))
                .init();
        }
        LogDestination::Cli => {
            // Console logging only
            tracing_subscriber::fmt()
                .with_timer(UnixTimestampFormatter)
                .with_target(false)
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
                )
                .init();
        }
        LogDestination::File => {
            // File logging only
            let (file_appender, log_dir) = if let Some(custom_path) = &args.log_file {
                // Use custom log file path
                let path = Path::new(custom_path);
                let log_dir = path.parent().unwrap_or(Path::new("."));
                let filename = path.file_name().unwrap().to_str().unwrap();
                
                if !log_dir.exists() {
                    std::fs::create_dir_all(log_dir)?;
                }
                (tracing_appender::rolling::daily(log_dir, filename), log_dir.display().to_string())
            } else {
                // Use default log path
                if !Path::new("logs").exists() {
                    std::fs::create_dir_all("logs")?;
                }
                (tracing_appender::rolling::daily("logs", "kokoros-http.log"), "logs".to_string())
            };
            
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            std::mem::forget(_guard); // Keep guard alive for the duration of the program
            
            tracing_subscriber::fmt()
                .with_timer(UnixTimestampFormatter)
                .with_writer(non_blocking)
                .with_target(false)
                .with_ansi(false)
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
                )
                .init();
            
            eprintln!("File logging enabled: {}", 
                if args.log_file.is_some() { args.log_file.as_ref().unwrap() } 
                else { "logs/kokoros-http.log" });
        }
        LogDestination::All => {
            // Both console and file logging
            let (file_appender, log_path) = if let Some(custom_path) = &args.log_file {
                // Use custom log file path
                let path = Path::new(custom_path);
                let log_dir = path.parent().unwrap_or(Path::new("."));
                let filename = path.file_name().unwrap().to_str().unwrap();
                
                if !log_dir.exists() {
                    std::fs::create_dir_all(log_dir)?;
                }
                (tracing_appender::rolling::daily(log_dir, filename), custom_path.clone())
            } else {
                // Use default log path
                if !Path::new("logs").exists() {
                    std::fs::create_dir_all("logs")?;
                }
                (tracing_appender::rolling::daily("logs", "kokoros-http.log"), "logs/kokoros-http.log".to_string())
            };
            
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            std::mem::forget(_guard); // Keep guard alive for the duration of the program
            
            use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
            tracing_subscriber::registry()
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_timer(UnixTimestampFormatter)
                        .with_target(false)
                )
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_timer(UnixTimestampFormatter)
                        .with_writer(non_blocking)
                        .with_target(false)
                        .with_ansi(false)
                )
                .with(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
                )
                .init();
            
            eprintln!("Console and file logging enabled: {}", log_path);
        }
    }

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
            instances,
            log_destination: _,
            log_file: _,
            mode,
        } = args;

        // Create TTS instance only for CLI modes, not for OpenAI server mode
        let tts = match &mode {
            Mode::OpenAI { .. } => None,
            _ => {
                // CLI modes always use single instance for optimal performance
                if instances > 1 {
                    tracing::info!("CLI mode: Using single instance for optimal performance (--instances {} ignored, WIP: to be supported in future)", instances);
                }
                Some(TTSKoko::new(&model_path, &data_path, 1).await)
            },
        };

        match mode {
            Mode::File {
                input_path,
                save_path_format,
            } => {
                let tts = tts.unwrap();
                let file_content = fs::read_to_string(input_path)?;
                for (i, line) in file_content.lines().enumerate() {
                    let stripped_line = line.trim();
                    if stripped_line.is_empty() {
                        continue;
                    }

                    let save_path = save_path_format.replace("{line}", &i.to_string());
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

            Mode::Text { text, save_path } => {
                let tts = tts.unwrap();
                let s = std::time::Instant::now();
                tts.tts(TTSOpts {
                    txt: &text,
                    lan: &lan,
                    style_name: &style,
                    save_path: &save_path,
                    mono,
                    speed,
                    initial_silence,
                })?;
                println!("Time taken: {:?}", s.elapsed());
                let words_per_second =
                    text.split_whitespace().count() as f32 / s.elapsed().as_secs_f32();
                println!("Words per second: {:.2}", words_per_second);
            }

            Mode::OpenAI { ip, port } => {
                // Warn about CPU performance with multiple instances
                #[cfg(not(feature = "cuda"))]
                if instances > 1 {
                    tracing::warn!("Multiple TTS instances ({}) on CPU may cause memory bandwidth contention", instances);
                    tracing::warn!("Consider using --instances 1 for optimal CPU performance");
                }

                // Create multiple independent TTS instances for parallel processing
                let mut tts_instances = Vec::new();
                for i in 0..instances {
                    tracing::info!("Initializing TTS instance [{}] ({}/{})", format!("{:02x}", i), i + 1, instances);
                    let instance = TTSKoko::new(&model_path, &data_path, instances).await;
                    tts_instances.push(instance);
                }
                let app = kokoros_openai::create_server(tts_instances, speed).await;
                let addr = SocketAddr::from((ip, port));
                let binding = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("Starting OpenAI-compatible HTTP server on {}", addr);
                tracing::info!("HTTP request/response logging enabled - logs saved to logs/kokoros-http.log");
                kokoros_openai::serve(binding, app.into_make_service()).await?;
            }

            Mode::Stream => {
                let tts = tts.unwrap();
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
                    match tts.tts_raw_audio(&stripped_line, &lan, &style, speed, initial_silence, None, None, None) {
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
