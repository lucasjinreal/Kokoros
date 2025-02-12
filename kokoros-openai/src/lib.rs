use std::error::Error;
use std::io;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::State, routing::post, Json, Router};
use kokoros::{
    tts::koko::TTSKoko,
    utils::wav::{write_audio_chunk, WavHeader},
};
use serde::Deserialize;
use tower_http::cors::CorsLayer;

#[derive(Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Wav,
}

#[derive(Deserialize)]
struct Voice(String);

impl Default for Voice {
    fn default() -> Self {
        Self("af_sky".into())
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

    // Must be WAV
    #[allow(dead_code)]
    #[serde(default)]
    response_format: AudioFormat,

    #[serde(default)]
    speed: Speed,

    #[serde(default)]
    initial_silence: Option<usize>,
}

pub async fn create_server(tts: TTSKoko) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(handle_tts))
        .layer(CorsLayer::permissive())
        .with_state(tts)
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
}

impl IntoResponse for SpeechError {
    fn into_response(self) -> Response {
        // None of these errors make sense to expose to the user of the API
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}

async fn handle_tts(
    State(tts): State<TTSKoko>,
    Json(SpeechRequest {
        model: _,
        input,
        voice: Voice(voice),
        response_format: _,
        speed: Speed(speed),
        initial_silence,
    }): Json<SpeechRequest>,
) -> Result<Vec<u8>, SpeechError> {
    let raw_audio = tts
        .tts_raw_audio(&input, "en-us", &voice, speed, initial_silence)
        .map_err(SpeechError::Koko)?;
    let mut wav_data = Vec::default();
    let header = WavHeader::new(1, TTSKoko::SAMPLE_RATE, 32);
    header
        .write_header(&mut wav_data)
        .map_err(SpeechError::Header)?;
    write_audio_chunk(&mut wav_data, &raw_audio).map_err(SpeechError::Chunk)?;

    Ok(wav_data)
}
