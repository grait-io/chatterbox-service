//! Configuration management for the TTS service

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the TTS service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Model configuration
    pub model: ModelConfig,
    /// Audio configuration
    pub audio: AudioConfig,
    /// Voice configuration
    pub voice: VoiceDefaults,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            audio: AudioConfig::default(),
            voice: VoiceDefaults::default(),
        }
    }
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Config::default();

        // Server config
        if let Ok(host) = std::env::var("HOST") {
            config.server.host = host;
        }
        if let Ok(port) = std::env::var("PORT") {
            if let Ok(p) = port.parse() {
                config.server.port = p;
            }
        }
        if let Ok(max_concurrent) = std::env::var("MAX_CONCURRENT_REQUESTS") {
            if let Ok(m) = max_concurrent.parse() {
                config.server.max_concurrent_requests = m;
            }
        }
        if let Ok(max_text) = std::env::var("MAX_TEXT_LENGTH") {
            if let Ok(m) = max_text.parse() {
                config.server.max_text_length = m;
            }
        }
        if let Ok(api_token) = std::env::var("API_TOKEN") {
            config.server.api_token = Some(api_token);
        }

        // Model config
        if let Ok(device) = std::env::var("DEVICE") {
            config.model.device = device;
        }
        if let Ok(load_standard) = std::env::var("LOAD_STANDARD") {
            config.model.load_standard = load_standard.to_lowercase() == "true";
        }
        if let Ok(load_turbo) = std::env::var("LOAD_TURBO") {
            config.model.load_turbo = load_turbo.to_lowercase() == "true";
        }
        if let Ok(load_multilingual) = std::env::var("LOAD_MULTILINGUAL") {
            config.model.load_multilingual = load_multilingual.to_lowercase() == "true";
        }
        if let Ok(cache_dir) = std::env::var("HF_HOME") {
            config.model.cache_dir = Some(PathBuf::from(cache_dir));
        }

        // Voice defaults
        if let Ok(voices_dir) = std::env::var("VOICES_DIR") {
            config.voice.voices_directory = PathBuf::from(voices_dir);
        }
        if let Ok(default_voice) = std::env::var("DEFAULT_VOICE") {
            config.voice.default_voice_path = Some(PathBuf::from(default_voice));
        }

        config
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Maximum text length in characters
    pub max_text_length: usize,
    /// API token for authentication (optional)
    pub api_token: Option<String>,
    /// Enable WebSocket streaming
    pub enable_websocket: bool,
    /// Enable HTTP endpoints
    pub enable_http: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8081,
            max_concurrent_requests: 3,
            max_text_length: 5000,
            api_token: None,
            enable_websocket: true,
            enable_http: true,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Device to run inference on (auto, cpu, cuda, mps)
    pub device: String,
    /// Load standard (expressive) model
    pub load_standard: bool,
    /// Load turbo (fast) model
    pub load_turbo: bool,
    /// Load multilingual model
    pub load_multilingual: bool,
    /// HuggingFace cache directory
    pub cache_dir: Option<PathBuf>,
    /// Warmup on startup
    pub warmup_on_startup: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            load_standard: true,
            load_turbo: false,
            load_multilingual: false,
            cache_dir: None,
            warmup_on_startup: true,
        }
    }
}

/// Audio output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Output sample rate
    pub sample_rate: u32,
    /// Audio format (pcm_s16le, opus)
    pub format: String,
    /// Chunk size in milliseconds for streaming
    pub chunk_size_ms: u32,
    /// Enable streaming output
    pub streaming: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            format: "pcm_s16le".to_string(),
            chunk_size_ms: 40,
            streaming: true,
        }
    }
}

/// Default voice settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceDefaults {
    /// Directory containing voice reference files
    pub voices_directory: PathBuf,
    /// Default voice reference file path
    pub default_voice_path: Option<PathBuf>,
    /// Default exaggeration (0.0-2.0)
    pub default_exaggeration: f32,
    /// Default cfg_weight (0.0-1.0)
    pub default_cfg_weight: f32,
    /// Default temperature (0.1-1.5)
    pub default_temperature: f32,
    /// Default top_p (0.0-1.0)
    pub default_top_p: f32,
    /// Default repetition penalty
    pub default_repetition_penalty: f32,
}

impl Default for VoiceDefaults {
    fn default() -> Self {
        Self {
            voices_directory: PathBuf::from("./voices"),
            default_voice_path: None,
            default_exaggeration: 0.5,
            default_cfg_weight: 0.5,
            default_temperature: 0.8,
            default_top_p: 0.95,
            default_repetition_penalty: 1.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 8081);
        assert_eq!(config.model.device, "auto");
        assert!(config.model.load_standard);
        assert!(!config.model.load_turbo);
    }
}
