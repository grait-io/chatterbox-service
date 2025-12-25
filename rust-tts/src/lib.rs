//! # Chatterbox TTS
//!
//! A high-performance Rust library for Text-to-Speech synthesis using
//! [Chatterbox](https://github.com/resemble-ai/chatterbox) models.
//!
//! ## Features
//!
//! - **Voice Cloning**: Clone any voice from a short audio sample (5-10 seconds)
//! - **Expressive Speech**: Control emotional intensity and pacing
//! - **Multiple Models**: Standard (expressive), Turbo (fast), Multilingual (23 languages)
//! - **Streaming**: Real-time audio streaming for low-latency applications
//! - **Easy Integration**: Simple API for embedding TTS in your Rust applications
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use chatterbox_tts::ChatterboxTts;
//!
//! fn main() -> chatterbox_tts::Result<()> {
//!     // Initialize the TTS engine
//!     let mut tts = ChatterboxTts::new()?;
//!
//!     // Load the model
//!     tts.load_model()?;
//!
//!     // Generate speech
//!     let audio = tts.synthesize("Hello, world!")?;
//!
//!     // Save to file
//!     audio.save_wav("output.wav")?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Voice Cloning
//!
//! ```rust,no_run
//! use chatterbox_tts::{ChatterboxTts, VoiceConfig};
//!
//! fn main() -> chatterbox_tts::Result<()> {
//!     let mut tts = ChatterboxTts::new()?;
//!     tts.load_model()?;
//!
//!     // Set a reference voice
//!     tts.set_voice("path/to/reference.wav")?;
//!
//!     // Generate with custom parameters
//!     let audio = tts
//!         .synthesize_with_config(
//!             "This will sound like the reference voice!",
//!             VoiceConfig::default()
//!                 .exaggeration(0.7)
//!                 .cfg_weight(0.5),
//!         )?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `server` - Enables HTTP/WebSocket server functionality
//! - `full` - Enables all features
//!

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core modules (always available)
pub mod audio;
pub mod config;
pub mod error;
mod models;
pub mod python;

// Server module (optional)
#[cfg(feature = "server")]
#[cfg_attr(docsrs, doc(cfg(feature = "server")))]
pub mod server;

// Re-exports for convenience
pub use error::{Result, TtsError};

// Main API types
pub use models::{
    AudioOutput, GenerationParams, ModelType, TtsModel, TtsModelManager, VoiceConfig,
};

// Audio utilities
pub use audio::{AudioBuffer, AudioChunk, AudioFormat};

// Configuration
pub use config::Config;

// Python utilities
pub use python::{detect_device, get_torch_info, initialize_python};

/// Prelude module for convenient imports
///
/// ```rust
/// use chatterbox_tts::prelude::*;
/// ```
pub mod prelude {
    //! Convenient re-exports for common usage
    pub use crate::audio::{AudioBuffer, AudioChunk, AudioFormat};
    pub use crate::config::Config;
    pub use crate::error::{Result, TtsError};
    pub use crate::models::{
        AudioOutput, GenerationParams, ModelType, TtsModel, TtsModelManager, VoiceConfig,
    };
    pub use crate::python::{detect_device, initialize_python};
    pub use crate::ChatterboxTts;
    pub use crate::ChatterboxTtsBuilder;
}

use std::path::Path;

/// High-level TTS interface for easy integration
///
/// This is the recommended way to use the library. It provides a simple,
/// ergonomic API for common TTS tasks.
///
/// # Example
///
/// ```rust,no_run
/// use chatterbox_tts::ChatterboxTts;
///
/// let mut tts = ChatterboxTts::new()?;
/// tts.load_model()?;
///
/// let audio = tts.synthesize("Hello!")?;
/// audio.save_wav("hello.wav")?;
/// # Ok::<(), chatterbox_tts::TtsError>(())
/// ```
pub struct ChatterboxTts {
    manager: TtsModelManager,
    default_voice: Option<std::path::PathBuf>,
    default_config: VoiceConfig,
    model_type: ModelType,
    device: String,
    initialized: bool,
}

impl ChatterboxTts {
    /// Create a new TTS instance with default settings
    ///
    /// This initializes the Python runtime and prepares for model loading.
    /// The model is not loaded until [`load_model`](Self::load_model) is called.
    pub fn new() -> Result<Self> {
        Self::with_config(Config::default())
    }

    /// Create a new TTS instance with custom configuration
    pub fn with_config(config: Config) -> Result<Self> {
        // Initialize Python runtime
        initialize_python(None)?;

        let device = if config.model.device == "auto" {
            detect_device()
        } else {
            config.model.device.clone()
        };

        Ok(Self {
            manager: TtsModelManager::new(),
            default_voice: config.voice.default_voice_path,
            default_config: VoiceConfig::default(),
            model_type: ModelType::Standard,
            device,
            initialized: false,
        })
    }

    /// Create a new TTS instance with a specific device
    ///
    /// # Arguments
    ///
    /// * `device` - The device to use: "auto", "cpu", "cuda", or "mps"
    pub fn with_device(device: impl Into<String>) -> Result<Self> {
        initialize_python(None)?;

        let device = device.into();
        let resolved_device = if device == "auto" {
            detect_device()
        } else {
            device
        };

        Ok(Self {
            manager: TtsModelManager::new(),
            default_voice: None,
            default_config: VoiceConfig::default(),
            model_type: ModelType::Standard,
            device: resolved_device,
            initialized: false,
        })
    }

    /// Get the device being used
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Check if models are loaded
    pub fn is_loaded(&self) -> bool {
        self.initialized
    }

    /// Load the default (Standard) model
    ///
    /// This downloads and loads the model weights. The first call may take
    /// some time as weights are downloaded from HuggingFace.
    pub fn load_model(&mut self) -> Result<()> {
        self.load_model_type(ModelType::Standard)
    }

    /// Load a specific model type
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of model to load
    pub fn load_model_type(&mut self, model_type: ModelType) -> Result<()> {
        tracing::info!("Loading {:?} model on {}...", model_type, self.device);

        self.manager.load_models(
            &self.device,
            model_type == ModelType::Standard,
            model_type == ModelType::Turbo,
            model_type == ModelType::Multilingual,
        )?;

        self.model_type = model_type;
        self.initialized = true;

        // Set default voice if configured
        if let Some(ref voice_path) = self.default_voice {
            if voice_path.exists() {
                self.manager.set_default_voice_path_all(voice_path)?;
            }
        }

        tracing::info!("Model loaded successfully");
        Ok(())
    }

    /// Load all available models
    pub fn load_all_models(&mut self) -> Result<()> {
        tracing::info!("Loading all models on {}...", self.device);

        self.manager.load_models(&self.device, true, true, true)?;
        self.initialized = true;

        if let Some(ref voice_path) = self.default_voice {
            if voice_path.exists() {
                self.manager.set_default_voice_path_all(voice_path)?;
            }
        }

        tracing::info!("All models loaded successfully");
        Ok(())
    }

    /// Warmup the loaded models
    ///
    /// This performs a short generation to ensure the model is fully loaded
    /// and ready for fast inference.
    pub fn warmup(&self) -> Result<()> {
        if !self.initialized {
            return Err(TtsError::ModelNotLoaded("No models loaded".into()));
        }
        self.manager.warmup()
    }

    /// Set the default voice from a file path
    ///
    /// The reference audio should be 5-10 seconds of clear speech.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the reference audio file (WAV, MP3, FLAC)
    pub fn set_voice<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TtsError::VoiceFileNotFound(path.display().to_string()));
        }

        self.default_voice = Some(path.to_path_buf());

        if self.initialized {
            self.manager.set_default_voice_path_all(path)?;
        }

        Ok(())
    }

    /// Clear the default voice
    pub fn clear_voice(&mut self) {
        self.default_voice = None;
    }

    /// Set the default voice configuration
    pub fn set_default_config(&mut self, config: VoiceConfig) {
        self.default_config = config;
    }

    /// Set the model type to use for synthesis
    pub fn set_model_type(&mut self, model_type: ModelType) {
        self.model_type = model_type;
    }

    /// Synthesize speech from text using default settings
    ///
    /// # Arguments
    ///
    /// * `text` - The text to synthesize
    ///
    /// # Returns
    ///
    /// Audio output containing the synthesized speech
    pub fn synthesize(&self, text: impl Into<String>) -> Result<AudioOutput> {
        self.synthesize_with_config(text, self.default_config.clone())
    }

    /// Synthesize speech with custom voice configuration
    ///
    /// # Arguments
    ///
    /// * `text` - The text to synthesize
    /// * `config` - Voice configuration parameters
    pub fn synthesize_with_config(
        &self,
        text: impl Into<String>,
        config: VoiceConfig,
    ) -> Result<AudioOutput> {
        if !self.initialized {
            return Err(TtsError::ModelNotLoaded("Model not loaded".into()));
        }

        let mut voice_config = config;

        // Use default voice if not specified
        if voice_config.audio_prompt_path.is_none() {
            voice_config.audio_prompt_path = self.default_voice.clone();
        }

        let params = GenerationParams::new(text)
            .voice(voice_config)
            .model_type(self.model_type);

        self.manager.generate(&params)
    }

    /// Synthesize multilingual speech
    ///
    /// # Arguments
    ///
    /// * `text` - The text to synthesize
    /// * `language` - Language code (e.g., "en", "es", "fr", "de", "zh", "ja")
    pub fn synthesize_multilingual(
        &self,
        text: impl Into<String>,
        language: impl Into<String>,
    ) -> Result<AudioOutput> {
        if !self.initialized {
            return Err(TtsError::ModelNotLoaded("Model not loaded".into()));
        }

        let mut voice_config = self.default_config.clone();
        if voice_config.audio_prompt_path.is_none() {
            voice_config.audio_prompt_path = self.default_voice.clone();
        }

        let params = GenerationParams::new(text)
            .voice(voice_config)
            .model_type(ModelType::Multilingual)
            .language(language);

        self.manager.generate(&params)
    }

    /// Get the underlying model manager for advanced usage
    pub fn manager(&self) -> &TtsModelManager {
        &self.manager
    }

    /// Get a mutable reference to the model manager
    pub fn manager_mut(&mut self) -> &mut TtsModelManager {
        &mut self.manager
    }
}

impl Default for ChatterboxTts {
    fn default() -> Self {
        Self::new().expect("Failed to create default ChatterboxTts instance")
    }
}

/// Builder for creating a [`ChatterboxTts`] instance with custom settings
///
/// # Example
///
/// ```rust,no_run
/// use chatterbox_tts::{ChatterboxTtsBuilder, ModelType, VoiceConfig};
///
/// let tts = ChatterboxTtsBuilder::new()
///     .device("cuda")
///     .model_type(ModelType::Turbo)
///     .voice("path/to/voice.wav")
///     .default_config(VoiceConfig::default().temperature(0.9))
///     .build()?;
/// # Ok::<(), chatterbox_tts::TtsError>(())
/// ```
pub struct ChatterboxTtsBuilder {
    device: String,
    model_type: ModelType,
    voice_path: Option<std::path::PathBuf>,
    default_config: VoiceConfig,
    auto_load: bool,
    warmup: bool,
}

impl ChatterboxTtsBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            device: "auto".to_string(),
            model_type: ModelType::Standard,
            voice_path: None,
            default_config: VoiceConfig::default(),
            auto_load: false,
            warmup: false,
        }
    }

    /// Set the device to use
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Set the model type
    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the default voice file
    pub fn voice<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.voice_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the default voice configuration
    pub fn default_config(mut self, config: VoiceConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Automatically load the model after building
    pub fn auto_load(mut self) -> Self {
        self.auto_load = true;
        self
    }

    /// Warmup the model after loading
    pub fn with_warmup(mut self) -> Self {
        self.warmup = true;
        self
    }

    /// Build the TTS instance
    pub fn build(self) -> Result<ChatterboxTts> {
        let mut tts = ChatterboxTts::with_device(&self.device)?;
        tts.model_type = self.model_type;
        tts.default_config = self.default_config;

        if let Some(voice_path) = self.voice_path {
            tts.default_voice = Some(voice_path);
        }

        if self.auto_load {
            tts.load_model_type(self.model_type)?;

            if self.warmup {
                tts.warmup()?;
            }
        }

        Ok(tts)
    }
}

impl Default for ChatterboxTtsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = ChatterboxTtsBuilder::new();
        assert_eq!(builder.device, "auto");
    }

    #[test]
    fn test_voice_config_builder() {
        let config = VoiceConfig::default()
            .exaggeration(0.8)
            .temperature(0.9);

        assert_eq!(config.exaggeration, 0.8);
        assert_eq!(config.temperature, 0.9);
    }
}
