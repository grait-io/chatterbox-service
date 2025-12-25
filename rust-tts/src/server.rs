//! HTTP and WebSocket server implementation

use crate::audio::{AudioBuffer, AudioFormat};
use crate::config::Config;
use crate::error::{Result, TtsError};
use crate::models::{GenerationParams, ModelType, TtsModelManager, VoiceConfig};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, Multipart, Path as AxumPath, Query, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

/// Shared application state
pub struct AppState {
    /// TTS model manager
    pub models: RwLock<TtsModelManager>,
    /// Configuration
    pub config: Config,
    /// Concurrency limiter
    pub semaphore: Semaphore,
    /// Voices directory
    pub voices_dir: PathBuf,
    /// Default voice path
    pub default_voice: RwLock<Option<PathBuf>>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: Config) -> Self {
        let semaphore = Semaphore::new(config.server.max_concurrent_requests);
        let voices_dir = config.voice.voices_directory.clone();
        let default_voice = config.voice.default_voice_path.clone();

        Self {
            models: RwLock::new(TtsModelManager::new()),
            config,
            semaphore,
            voices_dir,
            default_voice: RwLock::new(default_voice),
        }
    }
}

/// Server instance
pub struct Server {
    state: Arc<AppState>,
}

impl Server {
    /// Create a new server with the given configuration
    pub fn new(config: Config) -> Self {
        Self {
            state: Arc::new(AppState::new(config)),
        }
    }

    /// Initialize and load models
    pub fn load_models(&self) -> Result<()> {
        let config = &self.state.config;
        let device = if config.model.device == "auto" {
            crate::python::detect_device()
        } else {
            config.model.device.clone()
        };

        tracing::info!("Loading models on device: {}", device);

        let mut models = self.state.models.write();
        models.load_models(
            &device,
            config.model.load_standard,
            config.model.load_turbo,
            config.model.load_multilingual,
        )?;

        // Set default voice if configured
        if let Some(ref voice_path) = config.voice.default_voice_path {
            if voice_path.exists() {
                models.set_default_voice_path_all(voice_path)?;
                tracing::info!("Default voice set to: {}", voice_path.display());
            }
        }

        // Warmup if enabled
        if config.model.warmup_on_startup {
            models.warmup()?;
        }

        tracing::info!("Models loaded: {:?}", models.loaded_models());

        Ok(())
    }

    /// Set the default voice for all models
    pub fn set_default_voice(&self, voice_path: PathBuf) -> Result<()> {
        let models = self.state.models.read();
        models.set_default_voice_path_all(&voice_path)?;
        *self.state.default_voice.write() = Some(voice_path);
        Ok(())
    }

    /// Get the default voice path
    pub fn default_voice(&self) -> Option<PathBuf> {
        self.state.default_voice.read().clone()
    }

    /// Run the server
    pub async fn run(&self) -> Result<()> {
        let config = &self.state.config;
        let addr = format!("{}:{}", config.server.host, config.server.port);
        let socket_addr: SocketAddr = addr.parse().map_err(|e| {
            TtsError::Configuration(format!("Invalid address {}: {}", addr, e))
        })?;

        let app = self.build_router();

        tracing::info!("Starting server on {}", socket_addr);

        let listener = tokio::net::TcpListener::bind(socket_addr)
            .await
            .map_err(|e| TtsError::Io(e))?;

        axum::serve(listener, app)
            .await
            .map_err(|e| TtsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }

    /// Build the router
    fn build_router(&self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        Router::new()
            // Health check
            .route("/health", get(health_check))
            // HTTP TTS endpoint
            .route("/api/tts", post(generate_tts_http))
            // Set default voice
            .route("/api/voice/default", post(set_default_voice))
            // Get default voice
            .route("/api/voice/default", get(get_default_voice))
            // List available voices
            .route("/api/voices", get(list_voices))
            // Upload voice
            .route("/api/voices/upload", post(upload_voice))
            // WebSocket TTS endpoint
            .route("/ws/tts", get(websocket_handler))
            // Add state
            .with_state(self.state.clone())
            // Add middleware
            .layer(cors)
            .layer(TraceLayer::new_for_http())
    }
}

// ============================================================================
// API Types
// ============================================================================

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    loaded_models: Vec<String>,
    device: String,
    version: String,
}

/// TTS request
#[derive(Debug, Deserialize)]
struct TtsRequest {
    /// Text to synthesize
    text: String,
    /// Model type (standard, turbo, multilingual)
    #[serde(default)]
    model: Option<String>,
    /// Language ID for multilingual model
    #[serde(default)]
    language_id: Option<String>,
    /// Voice configuration
    #[serde(default)]
    voice: Option<VoiceParams>,
    /// Audio output configuration
    #[serde(default)]
    audio: Option<AudioParams>,
}

/// Voice parameters
#[derive(Debug, Default, Deserialize)]
struct VoiceParams {
    /// Reference audio file path or name
    #[serde(default)]
    audio_prompt_path: Option<String>,
    /// Exaggeration (0.0-2.0)
    #[serde(default)]
    exaggeration: Option<f32>,
    /// CFG weight (0.0-1.0)
    #[serde(default)]
    cfg_weight: Option<f32>,
    /// Temperature (0.1-1.5)
    #[serde(default)]
    temperature: Option<f32>,
    /// Top-p (0.0-1.0)
    #[serde(default)]
    top_p: Option<f32>,
    /// Repetition penalty
    #[serde(default)]
    repetition_penalty: Option<f32>,
}

/// Audio output parameters
#[derive(Debug, Default, Deserialize)]
struct AudioParams {
    /// Format (pcm_s16le, wav)
    #[serde(default)]
    format: Option<String>,
    /// Sample rate
    #[serde(default)]
    sample_rate: Option<u32>,
}

/// Set default voice request
#[derive(Debug, Deserialize)]
struct SetVoiceRequest {
    /// Voice file path or name in voices directory
    voice_path: String,
}

/// Voice info
#[derive(Serialize)]
struct VoiceInfo {
    name: String,
    path: String,
    duration_seconds: Option<f32>,
}

/// Error response
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
}

// ============================================================================
// HTTP Handlers
// ============================================================================

/// Health check endpoint
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let models = state.models.read();
    let loaded = models
        .loaded_models()
        .iter()
        .map(|m| m.to_string())
        .collect();

    Json(HealthResponse {
        status: "healthy".to_string(),
        loaded_models: loaded,
        device: models.device().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Generate TTS via HTTP
async fn generate_tts_http(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TtsRequest>,
) -> Response {
    // Acquire semaphore permit
    let _permit = match state.semaphore.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "Server is busy, please try again later".to_string(),
                    code: "SERVER_BUSY".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Validate text length
    if request.text.len() > state.config.server.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Text too long. Maximum length is {} characters",
                    state.config.server.max_text_length
                ),
                code: "TEXT_TOO_LONG".to_string(),
            }),
        )
            .into_response();
    }

    // Build voice config
    let defaults = &state.config.voice;
    let voice_params = request.voice.unwrap_or_default();

    let audio_prompt_path = voice_params.audio_prompt_path.map(|p| {
        // Check if it's a full path or just a name
        let path = PathBuf::from(&p);
        if path.is_absolute() && path.exists() {
            path
        } else {
            // Look in voices directory
            state.voices_dir.join(&p)
        }
    });

    let voice = VoiceConfig {
        audio_prompt_path,
        exaggeration: voice_params
            .exaggeration
            .unwrap_or(defaults.default_exaggeration),
        cfg_weight: voice_params
            .cfg_weight
            .unwrap_or(defaults.default_cfg_weight),
        temperature: voice_params
            .temperature
            .unwrap_or(defaults.default_temperature),
        top_p: voice_params.top_p.unwrap_or(defaults.default_top_p),
        repetition_penalty: voice_params
            .repetition_penalty
            .unwrap_or(defaults.default_repetition_penalty),
    };

    // Determine model type
    let model_type = match request.model.as_deref() {
        Some("turbo") => ModelType::Turbo,
        Some("multilingual") => ModelType::Multilingual,
        _ => ModelType::Standard,
    };

    // Build generation params
    let mut params = GenerationParams::new(&request.text)
        .voice(voice)
        .model_type(model_type);

    if let Some(lang) = request.language_id {
        params = params.language(lang);
    }

    // Generate audio
    let models = state.models.read();
    let result = tokio::task::spawn_blocking(move || models.generate(&params))
        .await
        .map_err(|e| TtsError::Inference(e.to_string()));

    let audio_output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(e)) | Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: "GENERATION_ERROR".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Determine output format
    let audio_params = request.audio.unwrap_or_default();
    let format = audio_params.format.as_deref().unwrap_or("wav");

    match format {
        "pcm_s16le" | "pcm" => {
            let pcm = audio_output.to_pcm_s16le();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "audio/pcm")],
                pcm,
            )
                .into_response()
        }
        _ => {
            let wav = crate::audio::samples_to_wav(&audio_output.samples, audio_output.sample_rate);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "audio/wav")],
                wav,
            )
                .into_response()
        }
    }
}

/// Set default voice
async fn set_default_voice(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SetVoiceRequest>,
) -> Response {
    let path = PathBuf::from(&request.voice_path);
    let full_path = if path.is_absolute() {
        path
    } else {
        state.voices_dir.join(&path)
    };

    if !full_path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Voice file not found: {}", full_path.display()),
                code: "VOICE_NOT_FOUND".to_string(),
            }),
        )
            .into_response();
    }

    let models = state.models.read();
    if let Err(e) = models.set_default_voice_path_all(&full_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "SET_VOICE_ERROR".to_string(),
            }),
        )
            .into_response();
    }

    *state.default_voice.write() = Some(full_path.clone());

    Json(serde_json::json!({
        "success": true,
        "voice_path": full_path.to_string_lossy()
    }))
    .into_response()
}

/// Get default voice
async fn get_default_voice(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let default_voice = state.default_voice.read().clone();
    Json(serde_json::json!({
        "default_voice": default_voice.map(|p| p.to_string_lossy().to_string())
    }))
}

/// List available voices
async fn list_voices(State(state): State<Arc<AppState>>) -> Response {
    let voices_dir = &state.voices_dir;

    if !voices_dir.exists() {
        return Json(serde_json::json!({
            "voices": [],
            "directory": voices_dir.to_string_lossy()
        }))
        .into_response();
    }

    let mut voices = Vec::new();

    if let Ok(entries) = std::fs::read_dir(voices_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "wav" || ext == "mp3" || ext == "flac" {
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    let duration = crate::python::get_audio_duration(path.to_str().unwrap_or("")).ok();

                    voices.push(VoiceInfo {
                        name,
                        path: path.to_string_lossy().to_string(),
                        duration_seconds: duration,
                    });
                }
            }
        }
    }

    Json(serde_json::json!({
        "voices": voices,
        "directory": voices_dir.to_string_lossy()
    }))
    .into_response()
}

/// Upload a voice file
async fn upload_voice(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            let filename = field.file_name().unwrap_or("voice.wav").to_string();
            let data = match field.bytes().await {
                Ok(d) => d,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: e.to_string(),
                            code: "UPLOAD_ERROR".to_string(),
                        }),
                    )
                        .into_response();
                }
            };

            // Ensure voices directory exists
            if let Err(e) = std::fs::create_dir_all(&state.voices_dir) {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                        code: "CREATE_DIR_ERROR".to_string(),
                    }),
                )
                    .into_response();
            }

            let path = state.voices_dir.join(&filename);
            if let Err(e) = std::fs::write(&path, &data) {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                        code: "WRITE_ERROR".to_string(),
                    }),
                )
                    .into_response();
            }

            return Json(serde_json::json!({
                "success": true,
                "filename": filename,
                "path": path.to_string_lossy()
            }))
            .into_response();
        }
    }

    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: "No file uploaded".to_string(),
            code: "NO_FILE".to_string(),
        }),
    )
        .into_response()
}

// ============================================================================
// WebSocket Handler
// ============================================================================

/// WebSocket upgrade handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: Arc<AppState>) {
    use futures_util::{SinkExt, StreamExt};
    let (mut sender, mut receiver) = socket.split();

    // Send ready message
    let models = state.models.read();
    let ready_msg = serde_json::json!({
        "type": "ready",
        "models": models.loaded_models().iter().map(|m| m.to_string()).collect::<Vec<_>>(),
        "device": models.device()
    });
    drop(models);

    if sender
        .send(Message::Text(ready_msg.to_string()))
        .await
        .is_err()
    {
        return;
    }

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(_) => break,
        };

        // Parse request
        let request: WsRequest = match serde_json::from_str(&msg) {
            Ok(r) => r,
            Err(e) => {
                let error_msg = serde_json::json!({
                    "type": "error",
                    "error": e.to_string(),
                    "code": "PARSE_ERROR"
                });
                let _ = sender.send(Message::Text(error_msg.to_string())).await;
                continue;
            }
        };

        match request.msg_type.as_str() {
            "say" => {
                handle_say_request(&mut sender, &state, request).await;
            }
            "set_voice" => {
                handle_set_voice_request(&mut sender, &state, request).await;
            }
            _ => {
                let error_msg = serde_json::json!({
                    "type": "error",
                    "error": format!("Unknown message type: {}", request.msg_type),
                    "code": "UNKNOWN_TYPE"
                });
                let _ = sender.send(Message::Text(error_msg.to_string())).await;
            }
        }
    }
}

/// WebSocket request
#[derive(Debug, Deserialize)]
struct WsRequest {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    language_id: Option<String>,
    #[serde(default)]
    voice: Option<VoiceParams>,
    #[serde(default)]
    audio: Option<WsAudioParams>,
    #[serde(default)]
    voice_path: Option<String>,
}

/// WebSocket audio parameters
#[derive(Debug, Default, Deserialize)]
struct WsAudioParams {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    chunk_size_ms: Option<u32>,
}

/// Handle "say" request
async fn handle_say_request(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    state: &Arc<AppState>,
    request: WsRequest,
) {
    use futures_util::SinkExt;

    let request_id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // Validate text
    let text = match request.text {
        Some(t) if !t.is_empty() => t,
        _ => {
            let error_msg = serde_json::json!({
                "type": "error",
                "id": request_id,
                "error": "Text is required",
                "code": "MISSING_TEXT"
            });
            let _ = sender.send(Message::Text(error_msg.to_string())).await;
            return;
        }
    };

    // Acquire semaphore
    let permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            let error_msg = serde_json::json!({
                "type": "error",
                "id": request_id,
                "error": "Server is busy",
                "code": "SERVER_BUSY"
            });
            let _ = sender.send(Message::Text(error_msg.to_string())).await;
            return;
        }
    };

    // Build voice config
    let defaults = &state.config.voice;
    let voice_params = request.voice.unwrap_or_default();

    let audio_prompt_path = voice_params.audio_prompt_path.map(|p| {
        let path = PathBuf::from(&p);
        if path.is_absolute() && path.exists() {
            path
        } else {
            state.voices_dir.join(&p)
        }
    });

    let voice = VoiceConfig {
        audio_prompt_path,
        exaggeration: voice_params
            .exaggeration
            .unwrap_or(defaults.default_exaggeration),
        cfg_weight: voice_params
            .cfg_weight
            .unwrap_or(defaults.default_cfg_weight),
        temperature: voice_params
            .temperature
            .unwrap_or(defaults.default_temperature),
        top_p: voice_params.top_p.unwrap_or(defaults.default_top_p),
        repetition_penalty: voice_params
            .repetition_penalty
            .unwrap_or(defaults.default_repetition_penalty),
    };

    // Determine model type
    let model_type = match request.model.as_deref() {
        Some("turbo") => ModelType::Turbo,
        Some("multilingual") => ModelType::Multilingual,
        _ => ModelType::Standard,
    };

    // Build generation params
    let mut params = GenerationParams::new(&text)
        .voice(voice)
        .model_type(model_type);

    if let Some(lang) = request.language_id {
        params = params.language(lang);
    }

    // Generate audio
    let models = state.models.read();
    let result = models.generate(&params);
    drop(models);
    drop(permit);

    let audio_output = match result {
        Ok(output) => output,
        Err(e) => {
            let error_msg = serde_json::json!({
                "type": "error",
                "id": request_id,
                "error": e.to_string(),
                "code": "GENERATION_ERROR"
            });
            let _ = sender.send(Message::Text(error_msg.to_string())).await;
            return;
        }
    };

    // Send audio chunks
    let audio_params = request.audio.unwrap_or_default();
    let chunk_size_ms = audio_params.chunk_size_ms.unwrap_or(40);
    let stream = audio_params.stream.unwrap_or(true);

    if stream {
        let chunks = audio_output.chunk_pcm(chunk_size_ms);
        for (i, chunk) in chunks.iter().enumerate() {
            let is_last = i == chunks.len() - 1;
            let chunk_msg = serde_json::json!({
                "type": "chunk",
                "id": request_id,
                "index": i,
                "data": base64::encode(chunk),
                "is_last": is_last
            });
            if sender.send(Message::Text(chunk_msg.to_string())).await.is_err() {
                return;
            }
        }
    } else {
        let pcm = audio_output.to_pcm_s16le();
        let audio_msg = serde_json::json!({
            "type": "audio",
            "id": request_id,
            "data": base64::encode(&pcm),
            "sample_rate": audio_output.sample_rate,
            "duration_seconds": audio_output.duration_seconds
        });
        if sender.send(Message::Text(audio_msg.to_string())).await.is_err() {
            return;
        }
    }

    // Send done message
    let done_msg = serde_json::json!({
        "type": "done",
        "id": request_id,
        "duration_seconds": audio_output.duration_seconds
    });
    let _ = sender.send(Message::Text(done_msg.to_string())).await;
}

/// Handle "set_voice" request
async fn handle_set_voice_request(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    state: &Arc<AppState>,
    request: WsRequest,
) {
    use futures_util::SinkExt;

    let voice_path = match request.voice_path {
        Some(p) => p,
        None => {
            let error_msg = serde_json::json!({
                "type": "error",
                "error": "voice_path is required",
                "code": "MISSING_VOICE_PATH"
            });
            let _ = sender.send(Message::Text(error_msg.to_string())).await;
            return;
        }
    };

    let full_path = {
        let path = PathBuf::from(&voice_path);
        if path.is_absolute() {
            path
        } else {
            state.voices_dir.join(&path)
        }
    };

    if !full_path.exists() {
        let error_msg = serde_json::json!({
            "type": "error",
            "error": format!("Voice file not found: {}", full_path.display()),
            "code": "VOICE_NOT_FOUND"
        });
        let _ = sender.send(Message::Text(error_msg.to_string())).await;
        return;
    }

    let models = state.models.read();
    if let Err(e) = models.set_default_voice_path_all(&full_path) {
        let error_msg = serde_json::json!({
            "type": "error",
            "error": e.to_string(),
            "code": "SET_VOICE_ERROR"
        });
        let _ = sender.send(Message::Text(error_msg.to_string())).await;
        return;
    }

    *state.default_voice.write() = Some(full_path.clone());

    let success_msg = serde_json::json!({
        "type": "voice_set",
        "voice_path": full_path.to_string_lossy()
    });
    let _ = sender.send(Message::Text(success_msg.to_string())).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_creation() {
        let config = Config::default();
        let state = AppState::new(config);
        assert!(state.default_voice.read().is_none());
    }
}
