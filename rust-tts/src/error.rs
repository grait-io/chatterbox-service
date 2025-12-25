//! Error types for the Chatterbox TTS service

use thiserror::Error;

/// Result type alias for TTS operations
pub type Result<T> = std::result::Result<T, TtsError>;

/// Errors that can occur during TTS operations
#[derive(Error, Debug)]
pub enum TtsError {
    /// Model loading failed
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Model not loaded
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Invalid voice reference file
    #[error("Invalid voice reference: {0}")]
    InvalidVoiceReference(String),

    /// Voice file not found
    #[error("Voice file not found: {0}")]
    VoiceFileNotFound(String),

    /// Text processing error
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),

    /// Audio processing error
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Python interop error
    #[error("Python error: {0}")]
    Python(String),

    /// WebSocket error
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Request timeout
    #[error("Request timeout")]
    Timeout,

    /// Service unavailable
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl From<pyo3::PyErr> for TtsError {
    fn from(err: pyo3::PyErr) -> Self {
        TtsError::Python(err.to_string())
    }
}
