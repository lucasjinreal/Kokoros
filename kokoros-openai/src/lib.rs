//! OpenAI-compatible TTS HTTP server for Kokoros
//!
//! This module provides an HTTP API that is compatible with OpenAI's text-to-speech endpoints.
//! It implements streaming and non-streaming audio generation with multiple format support.
//!
//! ## Implemented Features
//! - `/v1/audio/speech` - Text-to-speech generation with streaming support
//! - `/v1/audio/voices` - List available voices
//! - `/v1/models` - List available models (static dummy list)
//! - Multiple audio formats: MP3, WAV, PCM, OPUS, AAC, FLAC
//! - Streaming audio generation for low-latency responses
//!
//! ## OpenAI API Compatibility Limitations
//! - `return_download_link`: Not implemented (files are streamed directly)
//! - `lang_code`: Not implemented (language auto-detected from voice prefix)
//! - `volume_multiplier`: Not implemented (audio returned at original levels)
//! - `download_format`: Not implemented (only response_format used)
//! - `normalization_options`: Not implemented (basic text processing only)
//! - Streaming only supports PCM format (other formats fall back to PCM)

use std::error::Error;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::stream::StreamExt;
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::mp3::pcm_to_mp3,
    utils::opus::pcm_to_opus_ogg,
    utils::wav::{WavHeader, write_audio_chunk},
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

/// Break words used for chunk splitting
const BREAK_WORDS: &[&str] = &[
    "and", "or", "but", "&", "because", "if", "since", "though", "although", "however", "which",
];

/// Split text into speech chunks for streaming
///
/// Prioritizes sentence boundaries over word count for natural speech breaks
/// Then applies center-break word splitting for long chunks
fn split_text_into_speech_chunks(text: &str, words_per_chunk: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut word_count = 0;

    // First pass: split by punctuation
    for word in text.split_whitespace() {
        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        // Check for numbered list patterns: 1. 2) 3: (4), 5(\s)[.\)\:]
        let is_numbered_break = is_numbered_list_item(word);

        if is_numbered_break && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
            word_count = 0;
        }
        current_chunk.push_str(word);
        word_count += 1;

        // Check for unconditional breaks (always break regardless of word count)
        let ends_with_unconditional = word.ends_with('.')
            || word.ends_with('!')
            || word.ends_with('?')
            || word.ends_with(':')
            || word.ends_with(';');

        // Check for conditional breaks (commas - only break if enough words)
        let ends_with_conditional = word.ends_with(',');

        // Split conditions:
        // 1. Unconditional punctuation - always break
        // 2. Conditional punctuation + target word count reached
        if ends_with_unconditional
            || is_numbered_break
            || (ends_with_conditional && word_count >= words_per_chunk)
        {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
            word_count = 0;
        }
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Second pass: apply center-break splitting for long chunks
    // All chunks: ≥12 words
    // First 2 chunks: punctuation priority, Others: break words only
    let mut final_chunks = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let threshold = 12;
        let use_punctuation = index < 2; // First 2 chunks can use punctuation
        let split_chunks = split_long_chunk_with_depth(chunk, threshold, use_punctuation, 0);
        final_chunks.extend(split_chunks);
    }

    // Final processing: Move break words from end of chunks to beginning of next chunk
    for i in 0..final_chunks.len() - 1 {
        let current_chunk = &final_chunks[i];
        let words: Vec<&str> = current_chunk.trim().split_whitespace().collect();

        if let Some(last_word) = words.last() {
            // Check if last word is a break word (case insensitive)
            if BREAK_WORDS.contains(&last_word.to_lowercase().as_str()) && words.len() > 1 {
                // Only move if it won't create an empty chunk (need more than 1 word)
                let new_current = words[..words.len() - 1].join(" ");

                // Add break word to beginning of next chunk
                let next_chunk = &final_chunks[i + 1];
                let new_next = format!("{} {}", last_word, next_chunk);

                // Update the chunks
                final_chunks[i] = new_current;
                final_chunks[i + 1] = new_next;
            }
        }
    }

    // After all processing, there is no explicit filter to remove empty chunks.
    // If any empty string slipped through (e.g., from .trim().to_string() on
    // whitespace-only current_chunk, or from split_long_chunk), it would remain.
    // Dont consider filtering out empty chunks here, to enable catching potential bugs
    // in the chunking logic.
    final_chunks
}

/// Check if a word is a numbered list item: 1. 2) 3: (4), 5(\s)[.\)\:]
fn is_numbered_list_item(word: &str) -> bool {
    // Pattern matches: number followed by . ) or :
    // Examples: "1.", "2)", "3:", "(4)", "(5),"
    let numbered_regex = Regex::new(r"^\(?[0-9]+[.\)\:],?$").unwrap();
    numbered_regex.is_match(word)
}

fn split_long_chunk_with_depth(
    chunk: &str,
    threshold: usize,
    use_punctuation: bool,
    depth: usize,
) -> Vec<String> {
    // Prevent infinite recursion
    if depth >= 3 {
        return vec![chunk.to_string()];
    }
    let words: Vec<&str> = chunk.split_whitespace().collect();
    let word_count = words.len();

    // Only split if chunk meets the threshold
    if word_count < threshold {
        return vec![chunk.to_string()];
    }

    let center = word_count / 2;

    if use_punctuation {
        // Priority 1: Search for commas closest to center
        if let Some(pos) = find_closest_punctuation(&words, center, &[","]) {
            if pos >= 3 && pos < words.len() {
                let first_chunk = words[..pos].join(" ");
                let second_chunk = words[pos..].join(" ");

                // Recursively split both chunks if they're still too long
                let mut result = Vec::new();
                result.extend(split_long_chunk_with_depth(
                    &first_chunk,
                    threshold,
                    use_punctuation,
                    depth + 1,
                ));
                result.extend(split_long_chunk_with_depth(
                    &second_chunk,
                    threshold,
                    use_punctuation,
                    depth + 1,
                ));
                return result;
            }
        }
    }

    // Priority 2: Search for break words closest to center
    if let Some(pos) = find_closest_break_word(&words, center, BREAK_WORDS) {
        if pos >= 3 && pos < words.len() {
            let first_chunk = words[..pos].join(" ");
            let second_chunk = words[pos..].join(" ");

            // Recursively split both chunks if they're still too long
            let mut result = Vec::new();
            result.extend(split_long_chunk_with_depth(
                &first_chunk,
                threshold,
                use_punctuation,
                depth + 1,
            ));
            result.extend(split_long_chunk_with_depth(
                &second_chunk,
                threshold,
                use_punctuation,
                depth + 1,
            ));
            return result;
        }
    }

    // No suitable break point found, return original chunk
    vec![chunk.to_string()]
}

/// Find closest punctuation to center
fn find_closest_punctuation(words: &[&str], center: usize, punctuation: &[&str]) -> Option<usize> {
    let mut closest_pos = None;
    let mut min_distance = usize::MAX;

    for (i, word) in words.iter().enumerate() {
        if punctuation.iter().any(|p| word.ends_with(p)) {
            let distance = if i < center { center - i } else { i - center };
            if distance < min_distance {
                min_distance = distance;
                closest_pos = Some(i + 1); // Split after the punctuation
            }
        }
    }

    closest_pos
}

/// Find closest break word to center
fn find_closest_break_word(words: &[&str], center: usize, break_words: &[&str]) -> Option<usize> {
    let mut closest_pos = None;
    let mut min_distance = usize::MAX;

    for (i, word) in words.iter().enumerate() {
        if break_words.contains(&word.to_lowercase().as_str()) {
            let distance = if i < center { center - i } else { i - center };
            if distance < min_distance {
                min_distance = distance;
                closest_pos = Some(i); // Break word becomes first word of second chunk
            }
        }
    }

    closest_pos
}

#[derive(Deserialize, Default, Debug)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Mp3,
    Wav,
    Opus,
    Aac,
    Flac,
    Pcm,
}

/// Map OpenAI voice names to Kokoro voice names for backwards compatibility
///
/// OpenAI voices are mapped to their closest Kokoro equivalents:
/// - Direct matches: alloy → af_alloy, echo → am_echo, nova → af_nova, onyx → am_onyx
/// - Mapped voices: Other OpenAI voices are mapped to similar-sounding Kokoro voices
fn map_openai_voice_to_kokoro(voice: &str) -> &str {
    match voice.to_lowercase().as_str() {
        // Direct matches (same name exists in both)
        "alloy" => "af_alloy",
        "echo" => "am_echo",
        "nova" => "af_nova",
        "onyx" => "am_onyx",
        // Female OpenAI voices → Female Kokoro voices
        "shimmer" => "af_sky",
        "fable" => "af_bella",
        "coral" => "af_nicole",
        "sage" => "af_sarah",
        "marin" => "af_river",
        // Male OpenAI voices → Male Kokoro voices
        "ash" => "am_adam",
        "ballad" => "am_michael",
        "verse" => "am_eric",
        "cedar" => "am_liam",
        // If not an OpenAI voice, return as-is (assume it's already a Kokoro voice)
        _ => voice,
    }
}

#[derive(Deserialize)]
struct Voice(String);

impl Default for Voice {
    fn default() -> Self {
        Self("af_sky".into())
    }
}

impl Voice {
    /// Get the Kokoro voice name, mapping from OpenAI voice names if necessary
    fn to_kokoro_voice(&self) -> String {
        map_openai_voice_to_kokoro(&self.0).to_string()
    }
}

#[derive(Deserialize)]
struct Speed(f32);

impl Default for Speed {
    fn default() -> Self {
        Self(1.)
    }
}

#[derive(Deserialize)]
struct SpeechRequest {
    // Only one Kokoro model exists
    #[allow(dead_code)]
    model: String,

    input: String,

    #[serde(default)]
    voice: Voice,

    #[serde(default)]
    response_format: AudioFormat,

    #[serde(default)]
    speed: Speed,

    #[serde(default)]
    initial_silence: Option<usize>,

    /// Enable streaming audio generation (implemented)
    #[serde(default)]
    stream: Option<bool>,

    // OpenAI API compatibility parameters - accepted but not implemented
    // These fields ensure request parsing compatibility with OpenAI clients
    /// Return download link after generation (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    return_download_link: Option<bool>,

    /// Language code for text processing (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    lang_code: Option<String>,

    /// Volume multiplier for output audio (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    volume_multiplier: Option<f32>,

    /// Format for download when different from response_format (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    download_format: Option<String>,

    /// Text normalization options (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    normalization_options: Option<serde_json::Value>,
}

/// Async TTS worker task
#[derive(Debug)]
struct TTSTask {
    id: usize,
    chunk: String,
    voice: String,
    speed: f32,
    initial_silence: Option<usize>,
    result_tx: mpsc::UnboundedSender<(usize, Vec<u8>)>,
}

/// Streaming session manager
#[derive(Debug)]
struct StreamingSession {
    session_id: Uuid,
    start_time: Instant,
}

/// TTS worker pool manager with multiple TTS instances
#[derive(Clone)]
struct TTSWorkerPool {
    tts_instances: Vec<Arc<TTSKoko>>,
}

impl TTSWorkerPool {
    fn new(tts_instances: Vec<TTSKoko>) -> Self {
        Self {
            tts_instances: tts_instances.into_iter().map(Arc::new).collect(),
        }
    }

    fn get_instance(&self, worker_id: usize) -> (Arc<TTSKoko>, String) {
        let index = worker_id % self.tts_instances.len();
        let instance_id = format!("{:02x}", index);
        (Arc::clone(&self.tts_instances[index]), instance_id)
    }

    fn instance_count(&self) -> usize {
        self.tts_instances.len()
    }

    // process_chunk method removed - now handled inline in sequential queue processing
}

#[derive(Serialize)]
struct VoicesResponse {
    voices: Vec<String>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

pub async fn create_server(tts_instances: Vec<TTSKoko>) -> Router {
    info!("Starting TTS server with {} instances", tts_instances.len());

    // Use first instance for compatibility with non-streaming endpoints
    let tts_single = tts_instances
        .first()
        .cloned()
        .expect("At least one TTS instance required");

    Router::new()
        .route("/", get(handle_home))
        .route("/v1/audio/speech", post(handle_tts))
        .route("/v1/audio/voices", get(handle_voices))
        .route("/v1/models", get(handle_models))
        .route("/v1/models/{model}", get(handle_model))
        .layer(axum::middleware::from_fn(request_id_middleware))
        .layer(CorsLayer::permissive())
        .with_state((tts_single, tts_instances))
}

pub use axum::serve;

#[derive(Debug)]
enum SpeechError {
    // Deciding to modify this example in order to see errors
    // (e.g. with tracing) is up to the developer
    #[allow(dead_code)]
    Koko(Box<dyn Error>),

    #[allow(dead_code)]
    Header(io::Error),

    #[allow(dead_code)]
    Chunk(io::Error),

    #[allow(dead_code)]
    Mp3Conversion(std::io::Error),

    #[allow(dead_code)]
    OpusConversion(std::io::Error),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::Koko(e) => write!(f, "Koko TTS error: {}", e),
            SpeechError::Header(e) => write!(f, "Header error: {}", e),
            SpeechError::Chunk(e) => write!(f, "Chunk error: {}", e),
            SpeechError::Mp3Conversion(e) => write!(f, "MP3 conversion error: {}", e),
            SpeechError::OpusConversion(e) => write!(f, "Opus conversion error: {}", e),
        }
    }
}

impl IntoResponse for SpeechError {
    fn into_response(self) -> Response {
        // None of these errors make sense to expose to the user of the API
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}

/// Returns a 200 OK response to make it easier to check if the server is
/// running.
async fn handle_home() -> &'static str {
    "OK"
}

async fn handle_tts(
    State((tts_single, tts_instances)): State<(TTSKoko, Vec<TTSKoko>)>,
    request: axum::extract::Request,
) -> Result<Response, SpeechError> {
    let (request_id, request_start) = request
        .extensions()
        .get::<(String, Instant)>()
        .cloned()
        .unwrap_or_else(|| ("unknown".to_string(), Instant::now()));

    // OpenAI TTS always streams by default - client decides how to consume
    // Only send complete file when explicitly requested via stream: false

    // Parse the JSON body
    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| {
            error!("Error reading request body: {:?}", e);
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        })?;

    let speech_request: SpeechRequest = serde_json::from_slice(&bytes).map_err(|e| {
        error!("JSON parsing error: {:?}", e);
        SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
    })?;

    let SpeechRequest {
        input,
        voice,
        response_format,
        speed: Speed(speed),
        initial_silence,
        stream,
        ..
    } = speech_request;

    // Map OpenAI voice names to Kokoro voice names
    let voice = voice.to_kokoro_voice();

    // OpenAI-compliant behavior: Stream by default, only send complete file if stream: false
    let should_stream = stream.unwrap_or(false); // Default to not streaming

    let colored_request_id = get_colored_request_id_with_relative(&request_id, request_start);
    debug!(
        "{} Streaming decision: stream_param={:?}, final_decision={}",
        colored_request_id, stream, should_stream
    );

    if should_stream {
        return handle_tts_streaming(
            tts_instances,
            input,
            voice,
            response_format,
            speed,
            initial_silence,
            request_id,
            request_start,
        )
        .await;
    }

    // Non-streaming mode (existing implementation)
    let raw_audio = tts_single
        .tts_raw_audio(
            &input,
            "en-us",
            &voice,
            speed,
            initial_silence,
            Some(&request_id),
            Some("00"),
            None,
        )
        .map_err(SpeechError::Koko)?;

    let sample_rate = TTSKokoInitConfig::default().sample_rate;

    let (content_type, audio_data, format_name) = match response_format {
        AudioFormat::Wav => {
            let mut wav_data = Vec::default();
            let header = WavHeader::new(1, sample_rate, 32);
            header
                .write_header(&mut wav_data)
                .map_err(SpeechError::Header)?;
            write_audio_chunk(&mut wav_data, &raw_audio).map_err(SpeechError::Chunk)?;

            ("audio/wav", wav_data, "WAV")
        }
        AudioFormat::Opus => {
            let opus_data = pcm_to_opus_ogg(&raw_audio, sample_rate)
                .map_err(|e| SpeechError::OpusConversion(e))?;

            ("audio/opus", opus_data, "OPUS")
        }
        AudioFormat::Mp3 => {
            let mp3_data =
                pcm_to_mp3(&raw_audio, sample_rate).map_err(|e| SpeechError::Mp3Conversion(e))?;

            ("audio/mpeg", mp3_data, "MP3")
        }
        AudioFormat::Pcm => {
            // For PCM, we return the raw audio data directly
            // Convert f32 samples to 16-bit PCM
            let mut pcm_data = Vec::with_capacity(raw_audio.len() * 2);
            for sample in raw_audio {
                let pcm_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                pcm_data.extend_from_slice(&pcm_sample.to_le_bytes());
            }
            ("audio/pcm", pcm_data, "PCM")
        }
        // For now, unsupported formats fall back to MP3
        _ => {
            let mp3_data =
                pcm_to_mp3(&raw_audio, sample_rate).map_err(|e| SpeechError::Mp3Conversion(e))?;

            ("audio/mpeg", mp3_data, "MP3")
        }
    };

    let colored_request_id = get_colored_request_id_with_relative(&request_id, request_start);
    info!(
        "{} TTS non-streaming completed - {} bytes, {} format",
        colored_request_id,
        audio_data.len(),
        format_name
    );

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(audio_data.into())
        .map_err(|e| {
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?)
}

/// Handle streaming TTS requests with true async processing
///
/// Uses micro-chunking and parallel processing for low-latency streaming.
/// Maintains speech order while allowing out-of-order chunk completion.
async fn handle_tts_streaming(
    tts_instances: Vec<TTSKoko>,
    input: String,
    voice: String,
    response_format: AudioFormat,
    speed: f32,
    initial_silence: Option<usize>,
    request_id: String,
    request_start: Instant,
) -> Result<Response, SpeechError> {
    // Streaming implementation: PCM format for optimal performance
    let content_type = match response_format {
        AudioFormat::Pcm => "audio/pcm",
        _ => "audio/pcm", // Force PCM for optimal streaming performance
    };

    // Create worker pool with vector of TTS instances for true parallelism
    let worker_pool = TTSWorkerPool::new(tts_instances);

    // Create speech chunks based on word count and punctuation
    let mut chunks = split_text_into_speech_chunks(&input, 10);

    // Add empty chunk at end as completion signal to client
    chunks.push(String::new());
    let total_chunks = chunks.len();

    let colored_request_id = get_colored_request_id_with_relative(&request_id, request_start);
    debug!(
        "{} Processing {} chunks for streaming with window size {}",
        colored_request_id,
        total_chunks,
        worker_pool.instance_count()
    );

    if chunks.is_empty() {
        return Err(SpeechError::Mp3Conversion(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No text to process",
        )));
    }

    // Create channels for sequential chunk processing
    let (task_tx, mut task_rx) = mpsc::unbounded_channel::<TTSTask>();
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<(usize, Vec<u8>)>(); // Tag chunks with order ID

    // Track total bytes transferred
    let total_bytes = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Create session for tracking
    let session = StreamingSession {
        session_id: Uuid::new_v4(),
        start_time: Instant::now(),
    };

    let colored_request_id = get_colored_request_id_with_relative(&request_id, request_start);
    info!(
        "{} TTS session started - {} chunks streaming",
        colored_request_id, total_chunks
    );

    // Queue all tasks in order for sequential processing
    for (id, chunk) in chunks.into_iter().enumerate() {
        let task = TTSTask {
            id,
            chunk,
            voice: voice.clone(),
            speed,
            initial_silence: if id == 0 { initial_silence } else { None },
            result_tx: audio_tx.clone(),
        };

        task_tx.send(task).unwrap();
    }

    // Drop the task sender to signal completion
    drop(task_tx);

    // Windowed parallel processing: allow chunks to process concurrently up to available TTS instances
    let worker_pool_clone = worker_pool.clone();
    let total_bytes_clone = total_bytes.clone();
    let audio_tx_clone = audio_tx.clone();
    let total_chunks_expected = total_chunks;
    tokio::spawn(async move {
        use std::collections::BTreeMap;

        let mut chunk_counter = 0;
        let mut pending_chunks: BTreeMap<
            usize,
            tokio::task::JoinHandle<Result<(usize, Vec<u8>), String>>,
        > = BTreeMap::new();
        let mut next_to_send = 0;
        let mut chunks_processed = 0;
        let window_size = worker_pool_clone.instance_count(); // Allow chunks to process in parallel up to available TTS instances

        loop {
            // Receive new tasks while we have window space and tasks are available
            while pending_chunks.len() < window_size {
                // Use a non-blocking approach but with proper channel closure detection
                match task_rx.try_recv() {
                    Ok(task) => {
                        let task_id = task.id;
                        let worker_pool_clone = worker_pool_clone.clone();
                        let total_bytes_clone = total_bytes_clone.clone();
                        let request_id_clone = request_id.clone();

                        // Process chunk with dedicated TTS instance (alternates between instances)
                        let (tts_instance, actual_instance_id) =
                            worker_pool_clone.get_instance(chunk_counter);
                        let chunk_text = task.chunk.clone();
                        let voice = task.voice.clone();
                        let speed = task.speed;
                        let initial_silence = task.initial_silence;
                        let chunk_num = chunk_counter;

                        // Spawn parallel processing
                        let handle = tokio::spawn(async move {
                            // Handle empty chunks (completion signals) without TTS processing
                            if chunk_text.trim().is_empty() {
                                // Empty chunk - send as completion signal
                                return Ok((task_id, Vec::new()));
                            }

                            let result = tokio::task::spawn_blocking(move || {
                                let audio_result = tts_instance.tts_raw_audio(
                                    &chunk_text,
                                    "en-us",
                                    &voice,
                                    speed,
                                    initial_silence,
                                    Some(&request_id_clone),
                                    Some(&actual_instance_id),
                                    Some(chunk_num),
                                );

                                audio_result
                                    .map(|audio| audio)
                                    .map_err(|e| format!("TTS processing error: {:?}", e))
                            })
                            .await;

                            // Convert audio to PCM
                            match result {
                                Ok(Ok(audio_samples)) => {
                                    let mut pcm_data = Vec::with_capacity(audio_samples.len() * 2);
                                    for sample in audio_samples {
                                        let pcm_sample =
                                            (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                                        pcm_data.extend_from_slice(&pcm_sample.to_le_bytes());
                                    }
                                    total_bytes_clone.fetch_add(
                                        pcm_data.len(),
                                        std::sync::atomic::Ordering::Relaxed,
                                    );
                                    Ok((task_id, pcm_data))
                                }
                                Ok(Err(e)) => Err(e),
                                Err(e) => Err(format!("Task execution error: {:?}", e)),
                            }
                        });

                        pending_chunks.insert(chunk_counter, handle);
                        chunk_counter += 1;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        // No tasks available right now, break inner loop to check completed chunks
                        break;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        // Channel is closed, no more tasks will come
                        break;
                    }
                }
            }

            // Check if we can send the next chunk in order
            if let Some(handle) = pending_chunks.remove(&next_to_send) {
                if handle.is_finished() {
                    match handle.await {
                        Ok(Ok((task_id, pcm_data))) => {
                            if let Err(_) = audio_tx_clone.send((task_id, pcm_data)) {
                                break;
                            }
                            next_to_send += 1;
                            chunks_processed += 1;
                        }
                        Ok(Err(_e)) => {
                            // TTS processing error - skip this chunk
                            next_to_send += 1;
                            chunks_processed += 1;
                        }
                        Err(_e) => {
                            // Task execution error - skip this chunk
                            next_to_send += 1;
                            chunks_processed += 1;
                        }
                    }
                } else {
                    // Not finished yet, put it back
                    pending_chunks.insert(next_to_send, handle);
                }
            }

            // Check if all chunks have been processed and sent
            // We're done when we've processed all expected chunks
            if chunks_processed >= total_chunks_expected {
                break;
            }

            // Also check if we have no more work to do (fallback safety check)
            if pending_chunks.is_empty()
                && task_rx.is_empty()
                && chunks_processed < total_chunks_expected
            {
                // This shouldn't happen, but log it for debugging
                eprintln!(
                    "Warning: Early termination detected - processed {} of {} chunks",
                    chunks_processed, total_chunks_expected
                );
                break;
            }

            // Small delay to prevent busy waiting
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        // Wait for any remaining chunks to complete and collect them
        // This fixes the previous issue where only chunks matching next_to_send exactly were processed
        let mut remaining_chunks = Vec::new();

        for (chunk_id, handle) in pending_chunks {
            match handle.await {
                Ok(Ok((task_id, pcm_data))) => {
                    // Collect all successful chunks regardless of order
                    remaining_chunks.push((chunk_id, task_id, pcm_data));
                }
                Ok(Err(_e)) => {
                    // TTS processing error - still count as processed
                    chunks_processed += 1;
                }
                Err(_e) => {
                    // Task execution error - still count as processed
                    chunks_processed += 1;
                }
            }
        }

        // Sort remaining chunks by chunk_id to maintain proper order
        // This ensures audio continuity even for out-of-order completions
        remaining_chunks.sort_by_key(|(chunk_id, _, _)| *chunk_id);

        // Send all remaining chunks in order, preventing data loss
        for (chunk_id, task_id, pcm_data) in remaining_chunks {
            // Only send chunks that are in the expected sequence (>= next_to_send)
            // This prevents duplicate sends while ensuring no valid chunks are skipped
            if chunk_id >= next_to_send {
                let _ = audio_tx_clone.send((task_id, pcm_data));
                chunks_processed += 1;
            }
        }

        let _session_time = session.start_time.elapsed();

        // Log completion
        let bytes_transferred = total_bytes.load(std::sync::atomic::Ordering::Relaxed);
        // Calculate audio duration: 16-bit PCM (2 bytes per sample) at 24000 Hz
        let total_samples = bytes_transferred / 2;
        let duration_seconds = total_samples as f64 / 24000.0;
        let colored_request_id = get_colored_request_id_with_relative(&request_id, request_start);
        info!(
            "{} TTS session completed - {} chunks, {} bytes, {:.1}s audio, PCM format",
            colored_request_id, total_chunks, bytes_transferred, duration_seconds
        );

        // Send termination signal
        let _ = audio_tx.send((total_chunks, vec![])); // Empty data as termination signal
    });

    // No ordering needed - sequential processing guarantees order

    // Create immediate streaming - chunks are already sent in order from TTS processing
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(audio_rx)
        .map(|(_chunk_id, data)| -> Result<Vec<u8>, std::io::Error> {
            // Check for termination signal (empty data)
            if data.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Stream complete",
                ));
            }
            Ok(data)
        })
        .take_while(|result| {
            // Continue until we hit an error (termination signal)
            std::future::ready(result.is_ok())
        });

    // Convert to HTTP body with explicit ordering
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONNECTION, "keep-alive")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no") // Disable nginx buffering
        .header("Transfer-Encoding", "chunked") // Enable HTTP chunked transfer encoding
        .header("Access-Control-Allow-Origin", "*") // CORS for browser clients
        .body(body)
        .map_err(|e| {
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?)
}

async fn handle_voices(
    State((tts_single, _tts_instances)): State<(TTSKoko, Vec<TTSKoko>)>,
) -> Json<VoicesResponse> {
    let mut voices = tts_single.get_available_voices();

    // Add OpenAI voice names for compatibility
    let openai_voices = vec![
        "alloy".to_string(),
        "echo".to_string(),
        "nova".to_string(),
        "onyx".to_string(),
        "shimmer".to_string(),
        "fable".to_string(),
        "coral".to_string(),
        "sage".to_string(),
        "marin".to_string(),
        "ash".to_string(),
        "ballad".to_string(),
        "verse".to_string(),
        "cedar".to_string(),
    ];

    voices.extend(openai_voices);
    Json(VoicesResponse { voices })
}

/// Handle /v1/models endpoint
///
/// Returns a static list of models for OpenAI API compatibility.
/// Note: All models use the same underlying Kokoro TTS engine.
async fn handle_models() -> Json<ModelsResponse> {
    let models = vec![
        ModelObject {
            id: "tts-1".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        ModelObject {
            id: "tts-1-hd".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        ModelObject {
            id: "kokoro".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        ModelObject {
            id: "gpt-4o-mini-tts".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
    ];

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

async fn handle_model(Path(model_id): Path<String>) -> Result<Json<ModelObject>, StatusCode> {
    let model = match model_id.as_str() {
        "tts-1" => ModelObject {
            id: "tts-1".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        "tts-1-hd" => ModelObject {
            id: "tts-1-hd".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        "kokoro" => ModelObject {
            id: "kokoro".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        "gpt-4o-mini-tts" => ModelObject {
            id: "tts-1-hd".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        _ => return Err(StatusCode::NOT_FOUND),
    };

    Ok(Json(model))
}

fn get_colored_request_id_with_relative(request_id: &str, start_time: Instant) -> String {
    kokoros::utils::debug::get_colored_request_id_with_relative(request_id, start_time)
}

async fn request_id_middleware(
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = request.method().clone();
    let uri = request.uri().path().to_string();
    let user_agent = request
        .headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("-")
        .to_string();

    let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let start = std::time::Instant::now();
    let colored_request_id = get_colored_request_id_with_relative(&request_id, start);
    request.extensions_mut().insert((request_id.clone(), start));

    info!(
        "{} {} {} \"{}\"",
        colored_request_id, method, uri, user_agent
    );

    let response = next.run(request).await;
    let _latency = start.elapsed();

    let colored_request_id_response = get_colored_request_id_with_relative(&request_id, start);
    info!("{} {}", colored_request_id_response, response.status());

    response
}
