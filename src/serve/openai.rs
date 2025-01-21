use crate::tts::koko::TTSKoko;
use axum::http::StatusCode;
use axum::{extract::State, routing::post, Json, Router};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

#[derive(Deserialize)]
struct TTSRequest {
    model: String,
    input: String,
    voice: Option<String>,
    return_audio: Option<bool>,
}

#[derive(Serialize)]
struct TTSResponse {
    status: String,
    file_path: Option<String>, // Made optional since we won't always have a file
    audio: Option<String>,     // Made optional since we won't always return audio
}

pub async fn create_server(tts: TTSKoko) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(handle_tts))
        .layer(CorsLayer::permissive())
        .with_state(tts)
}

async fn handle_tts(
    State(tts): State<TTSKoko>,
    Json(payload): Json<TTSRequest>,
) -> Result<Json<TTSResponse>, StatusCode> {
    let voice = payload.voice.unwrap_or_else(|| "af_sky".to_string());
    let return_audio = payload.return_audio.unwrap_or(false);

    match tts.tts_raw_audio(&payload.input, "en-us", &voice) {
        Ok((audio_data, raw_audio)) => {
            if return_audio {
                let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&audio_data);
                Ok(Json(TTSResponse {
                    status: "success".to_string(),
                    file_path: None,
                    audio: Some(audio_base64),
                }))
            } else {
                let output_path = format!(
                    "tmp/output_{}.wav",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                );

                // Create WAV file
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000, // Using the same sample rate as in TTSKoko
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };

                if let Ok(mut writer) = hound::WavWriter::create(&output_path, spec) {
                    for &sample in &raw_audio {
                        if writer.write_sample(sample).is_err() {
                            return Err(StatusCode::INTERNAL_SERVER_ERROR);
                        }
                    }
                    if writer.finalize().is_err() {
                        return Err(StatusCode::INTERNAL_SERVER_ERROR);
                    }

                    Ok(Json(TTSResponse {
                        status: "success".to_string(),
                        file_path: Some(output_path),
                        audio: None,
                    }))
                } else {
                    Err(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}
