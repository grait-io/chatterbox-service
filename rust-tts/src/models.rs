//! TTS model definitions and inference

use crate::error::{Result, TtsError};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Type of TTS model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Standard model with expressiveness support
    Standard,
    /// Turbo model for fast inference
    Turbo,
    /// Multilingual model
    Multilingual,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Standard => write!(f, "standard"),
            ModelType::Turbo => write!(f, "turbo"),
            ModelType::Multilingual => write!(f, "multilingual"),
        }
    }
}

/// Voice configuration for TTS generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Path to reference audio file for voice cloning
    pub audio_prompt_path: Option<PathBuf>,
    /// Emotional exaggeration (0.0-2.0, standard model only)
    pub exaggeration: f32,
    /// Classifier-free guidance weight (0.0-1.0, standard model only)
    pub cfg_weight: f32,
    /// Sampling temperature (0.1-1.5)
    pub temperature: f32,
    /// Top-p sampling (0.0-1.0)
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            audio_prompt_path: None,
            exaggeration: 0.5,
            cfg_weight: 0.5,
            temperature: 0.8,
            top_p: 0.95,
            repetition_penalty: 1.2,
        }
    }
}

impl VoiceConfig {
    /// Create a new voice config with a reference audio file
    pub fn with_voice<P: AsRef<Path>>(voice_path: P) -> Self {
        Self {
            audio_prompt_path: Some(voice_path.as_ref().to_path_buf()),
            ..Default::default()
        }
    }

    /// Set the exaggeration level
    pub fn exaggeration(mut self, value: f32) -> Self {
        self.exaggeration = value.clamp(0.0, 2.0);
        self
    }

    /// Set the cfg_weight
    pub fn cfg_weight(mut self, value: f32) -> Self {
        self.cfg_weight = value.clamp(0.0, 1.0);
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, value: f32) -> Self {
        self.temperature = value.clamp(0.1, 1.5);
        self
    }

    /// Set the top_p value
    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = value.clamp(0.0, 1.0);
        self
    }

    /// Set the repetition penalty
    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = value.max(1.0);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(ref path) = self.audio_prompt_path {
            if !path.exists() {
                return Err(TtsError::VoiceFileNotFound(path.display().to_string()));
            }
        }
        Ok(())
    }
}

/// Parameters for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Text to synthesize
    pub text: String,
    /// Voice configuration
    pub voice: VoiceConfig,
    /// Model type to use
    pub model_type: ModelType,
    /// Language ID (for multilingual model)
    pub language_id: Option<String>,
}

impl GenerationParams {
    /// Create new generation parameters
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: VoiceConfig::default(),
            model_type: ModelType::Standard,
            language_id: None,
        }
    }

    /// Set the voice configuration
    pub fn voice(mut self, voice: VoiceConfig) -> Self {
        self.voice = voice;
        self
    }

    /// Set the model type
    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the language ID
    pub fn language(mut self, language_id: impl Into<String>) -> Self {
        self.language_id = Some(language_id.into());
        self
    }
}

/// Audio output from TTS generation
///
/// Contains the synthesized audio samples along with metadata.
/// Provides methods for saving to files and streaming.
///
/// # Example
///
/// ```rust,no_run
/// use chatterbox_tts::AudioOutput;
///
/// let audio = AudioOutput::new(vec![0.0; 24000], 24000);
/// audio.save_wav("output.wav")?;
/// # Ok::<(), chatterbox_tts::TtsError>(())
/// ```
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Raw audio samples (24kHz, mono, f32)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration_seconds: f32,
}

impl AudioOutput {
    /// Create a new audio output
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        let duration_seconds = samples.len() as f32 / sample_rate as f32;
        Self {
            samples,
            sample_rate,
            duration_seconds,
        }
    }

    /// Save audio to a WAV file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output WAV file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # let audio = chatterbox_tts::AudioOutput::new(vec![0.0; 24000], 24000);
    /// audio.save_wav("speech.wav")?;
    /// # Ok::<(), chatterbox_tts::TtsError>(())
    /// ```
    pub fn save_wav<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| TtsError::AudioProcessing(format!("Failed to create WAV file: {}", e)))?;

        for sample in &self.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let int_sample = (clamped * 32767.0) as i16;
            writer.write_sample(int_sample)
                .map_err(|e| TtsError::AudioProcessing(format!("Failed to write sample: {}", e)))?;
        }

        writer.finalize()
            .map_err(|e| TtsError::AudioProcessing(format!("Failed to finalize WAV: {}", e)))?;

        Ok(())
    }

    /// Save audio to a raw PCM file (S16LE format)
    pub fn save_pcm<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let pcm = self.to_pcm_s16le();
        std::fs::write(path, pcm)?;
        Ok(())
    }

    /// Convert to WAV bytes (in-memory)
    pub fn to_wav(&self) -> Vec<u8> {
        crate::audio::samples_to_wav(&self.samples, self.sample_rate)
    }

    /// Convert to PCM S16LE bytes
    pub fn to_pcm_s16le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        for sample in &self.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let int_sample = (clamped * 32767.0) as i16;
            bytes.extend_from_slice(&int_sample.to_le_bytes());
        }
        bytes
    }

    /// Chunk audio into fixed-size pieces for streaming
    ///
    /// # Arguments
    ///
    /// * `chunk_size_ms` - Size of each chunk in milliseconds
    pub fn chunk(&self, chunk_size_ms: u32) -> Vec<Vec<f32>> {
        let samples_per_chunk = (self.sample_rate * chunk_size_ms / 1000) as usize;
        self.samples
            .chunks(samples_per_chunk)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Chunk audio into PCM S16LE bytes for streaming
    ///
    /// # Arguments
    ///
    /// * `chunk_size_ms` - Size of each chunk in milliseconds
    pub fn chunk_pcm(&self, chunk_size_ms: u32) -> Vec<Vec<u8>> {
        let samples_per_chunk = (self.sample_rate * chunk_size_ms / 1000) as usize;
        self.samples
            .chunks(samples_per_chunk)
            .map(|chunk| {
                let mut bytes = Vec::with_capacity(chunk.len() * 2);
                for sample in chunk {
                    let clamped = sample.clamp(-1.0, 1.0);
                    let int_sample = (clamped * 32767.0) as i16;
                    bytes.extend_from_slice(&int_sample.to_le_bytes());
                }
                bytes
            })
            .collect()
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the audio is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// TTS Model wrapper that interfaces with Python
pub struct TtsModel {
    /// Python TTS model object
    py_model: Arc<RwLock<PyObject>>,
    /// Model type
    model_type: ModelType,
    /// Device the model is running on
    device: String,
    /// Whether the model is loaded
    loaded: bool,
    /// Default voice configuration
    default_voice: RwLock<VoiceConfig>,
}

impl TtsModel {
    /// Create a new TTS model (unloaded)
    pub fn new(model_type: ModelType) -> Self {
        Self {
            py_model: Arc::new(RwLock::new(Python::with_gil(|py| py.None()))),
            model_type,
            device: "cpu".to_string(),
            loaded: false,
            default_voice: RwLock::new(VoiceConfig::default()),
        }
    }

    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the device
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Set the default voice configuration
    pub fn set_default_voice(&self, voice: VoiceConfig) {
        *self.default_voice.write() = voice;
    }

    /// Get the default voice configuration
    pub fn default_voice(&self) -> VoiceConfig {
        self.default_voice.read().clone()
    }

    /// Set the default voice from a file path
    pub fn set_default_voice_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TtsError::VoiceFileNotFound(path.display().to_string()));
        }
        let mut voice = self.default_voice.write();
        voice.audio_prompt_path = Some(path.to_path_buf());
        Ok(())
    }

    /// Load the model
    pub fn load(&mut self, device: &str) -> Result<()> {
        let result: std::result::Result<(), PyErr> = Python::with_gil(|py| {
            // Import the appropriate TTS class
            let tts_module = match self.model_type {
                ModelType::Standard => py.import("chatterbox.tts")?,
                ModelType::Turbo => py.import("chatterbox.tts_turbo")?,
                ModelType::Multilingual => py.import("chatterbox.mtl_tts")?,
            };

            // Get the TTS class
            let tts_class = match self.model_type {
                ModelType::Standard => tts_module.getattr("ChatterboxTTS")?,
                ModelType::Turbo => tts_module.getattr("ChatterboxTurboTTS")?,
                ModelType::Multilingual => tts_module.getattr("ChatterboxMultilingualTTS")?,
            };

            // Create the model instance
            let kwargs = PyDict::new(py);
            kwargs.set_item("device", device)?;

            let model = tts_class.call_method("from_pretrained", (), Some(&kwargs))?;

            // Store the model
            *self.py_model.write() = model.into_py(py);
            Ok(())
        });

        match result {
            Ok(()) => {
                self.device = device.to_string();
                self.loaded = true;
                tracing::info!(
                    "Loaded {} model on {}",
                    self.model_type,
                    self.device
                );
                Ok(())
            }
            Err(e) => Err(TtsError::ModelLoad(e.to_string())),
        }
    }

    /// Warmup the model with a test generation
    pub fn warmup(&self) -> Result<()> {
        if !self.loaded {
            return Err(TtsError::ModelNotLoaded(self.model_type.to_string()));
        }

        tracing::info!("Warming up {} model...", self.model_type);

        let params = GenerationParams::new("Hello.")
            .voice(VoiceConfig::default())
            .model_type(self.model_type);

        // Generate a short sample to warm up the model
        let _ = self.generate(&params)?;

        tracing::info!("{} model warmup complete", self.model_type);
        Ok(())
    }

    /// Generate audio from text
    pub fn generate(&self, params: &GenerationParams) -> Result<AudioOutput> {
        if !self.loaded {
            return Err(TtsError::ModelNotLoaded(self.model_type.to_string()));
        }

        // Use default voice if none specified
        let voice = if params.voice.audio_prompt_path.is_some() {
            params.voice.clone()
        } else {
            let default = self.default_voice.read().clone();
            VoiceConfig {
                audio_prompt_path: default.audio_prompt_path,
                exaggeration: params.voice.exaggeration,
                cfg_weight: params.voice.cfg_weight,
                temperature: params.voice.temperature,
                top_p: params.voice.top_p,
                repetition_penalty: params.voice.repetition_penalty,
            }
        };

        let result: std::result::Result<Vec<f32>, PyErr> = Python::with_gil(|py| {
            let model = self.py_model.read();
            let model = model.bind(py);

            // Build kwargs based on model type
            let kwargs = PyDict::new(py);
            kwargs.set_item("text", &params.text)?;

            // Add voice reference if available
            if let Some(ref audio_path) = voice.audio_prompt_path {
                kwargs.set_item("audio_prompt_path", audio_path.to_str().unwrap_or(""))?;
            }

            // Model-specific parameters
            match self.model_type {
                ModelType::Standard => {
                    kwargs.set_item("exaggeration", voice.exaggeration)?;
                    kwargs.set_item("cfg_weight", voice.cfg_weight)?;
                }
                ModelType::Turbo => {
                    // Turbo doesn't support exaggeration/cfg_weight
                }
                ModelType::Multilingual => {
                    if let Some(ref lang) = params.language_id {
                        kwargs.set_item("language_id", lang)?;
                    }
                    kwargs.set_item("exaggeration", voice.exaggeration)?;
                    kwargs.set_item("cfg_weight", voice.cfg_weight)?;
                }
            }

            // Common parameters
            kwargs.set_item("temperature", voice.temperature)?;
            kwargs.set_item("top_p", voice.top_p)?;
            kwargs.set_item("repetition_penalty", voice.repetition_penalty)?;

            // Generate audio
            let result = model.call_method("generate", (), Some(&kwargs))?;

            // Convert numpy array to Vec<f32>
            let numpy = py.import("numpy")?;
            let wav = result.getattr("wav")?;
            let sr = result.getattr("sr")?;

            // Flatten and convert to list
            let wav_flat = wav.call_method0("flatten")?;
            let wav_list = wav_flat.call_method0("tolist")?;
            let wav_list = wav_list.downcast::<PyList>()?;

            let samples: Vec<f32> = wav_list
                .iter()
                .map(|x| x.extract::<f32>().unwrap_or(0.0))
                .collect();

            Ok(samples)
        });

        match result {
            Ok(samples) => Ok(AudioOutput::new(samples, 24000)),
            Err(e) => Err(TtsError::Inference(e.to_string())),
        }
    }
}

/// Container for multiple TTS models
pub struct TtsModelManager {
    /// Standard (expressive) model
    standard: Option<TtsModel>,
    /// Turbo (fast) model
    turbo: Option<TtsModel>,
    /// Multilingual model
    multilingual: Option<TtsModel>,
    /// Device being used
    device: String,
}

impl TtsModelManager {
    /// Create a new model manager
    pub fn new() -> Self {
        Self {
            standard: None,
            turbo: None,
            multilingual: None,
            device: "cpu".to_string(),
        }
    }

    /// Get the device being used
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Load models based on configuration
    pub fn load_models(
        &mut self,
        device: &str,
        load_standard: bool,
        load_turbo: bool,
        load_multilingual: bool,
    ) -> Result<()> {
        self.device = device.to_string();

        if load_standard {
            let mut model = TtsModel::new(ModelType::Standard);
            model.load(device)?;
            self.standard = Some(model);
        }

        if load_turbo {
            let mut model = TtsModel::new(ModelType::Turbo);
            model.load(device)?;
            self.turbo = Some(model);
        }

        if load_multilingual {
            let mut model = TtsModel::new(ModelType::Multilingual);
            model.load(device)?;
            self.multilingual = Some(model);
        }

        Ok(())
    }

    /// Warmup all loaded models
    pub fn warmup(&self) -> Result<()> {
        if let Some(ref model) = self.standard {
            model.warmup()?;
        }
        if let Some(ref model) = self.turbo {
            model.warmup()?;
        }
        if let Some(ref model) = self.multilingual {
            model.warmup()?;
        }
        Ok(())
    }

    /// Get a reference to the standard model
    pub fn standard(&self) -> Option<&TtsModel> {
        self.standard.as_ref()
    }

    /// Get a reference to the turbo model
    pub fn turbo(&self) -> Option<&TtsModel> {
        self.turbo.as_ref()
    }

    /// Get a reference to the multilingual model
    pub fn multilingual(&self) -> Option<&TtsModel> {
        self.multilingual.as_ref()
    }

    /// Get a model by type
    pub fn get_model(&self, model_type: ModelType) -> Option<&TtsModel> {
        match model_type {
            ModelType::Standard => self.standard(),
            ModelType::Turbo => self.turbo(),
            ModelType::Multilingual => self.multilingual(),
        }
    }

    /// Get the preferred model (standard if available, otherwise turbo)
    pub fn preferred_model(&self) -> Option<&TtsModel> {
        self.standard()
            .or_else(|| self.turbo())
            .or_else(|| self.multilingual())
    }

    /// Generate audio using the appropriate model
    pub fn generate(&self, params: &GenerationParams) -> Result<AudioOutput> {
        let model = self
            .get_model(params.model_type)
            .or_else(|| self.preferred_model())
            .ok_or_else(|| TtsError::ModelNotLoaded("No models loaded".to_string()))?;

        model.generate(params)
    }

    /// Set default voice for a specific model type
    pub fn set_default_voice(&self, model_type: ModelType, voice: VoiceConfig) -> Result<()> {
        let model = self
            .get_model(model_type)
            .ok_or_else(|| TtsError::ModelNotLoaded(model_type.to_string()))?;

        model.set_default_voice(voice);
        Ok(())
    }

    /// Set default voice path for all loaded models
    pub fn set_default_voice_path_all<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        if let Some(ref model) = self.standard {
            model.set_default_voice_path(path)?;
        }
        if let Some(ref model) = self.turbo {
            model.set_default_voice_path(path)?;
        }
        if let Some(ref model) = self.multilingual {
            model.set_default_voice_path(path)?;
        }

        Ok(())
    }

    /// Get list of loaded models
    pub fn loaded_models(&self) -> Vec<ModelType> {
        let mut models = Vec::new();
        if self.standard.is_some() {
            models.push(ModelType::Standard);
        }
        if self.turbo.is_some() {
            models.push(ModelType::Turbo);
        }
        if self.multilingual.is_some() {
            models.push(ModelType::Multilingual);
        }
        models
    }
}

impl Default for TtsModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_config_defaults() {
        let config = VoiceConfig::default();
        assert_eq!(config.exaggeration, 0.5);
        assert_eq!(config.cfg_weight, 0.5);
        assert_eq!(config.temperature, 0.8);
    }

    #[test]
    fn test_voice_config_builder() {
        let config = VoiceConfig::default()
            .exaggeration(1.5)
            .cfg_weight(0.7)
            .temperature(0.5);

        assert_eq!(config.exaggeration, 1.5);
        assert_eq!(config.cfg_weight, 0.7);
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_audio_output_pcm() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let output = AudioOutput::new(samples, 24000);

        let pcm = output.to_pcm_s16le();
        assert_eq!(pcm.len(), 10); // 5 samples * 2 bytes each
    }

    #[test]
    fn test_audio_chunking() {
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 / 4800.0)).collect();
        let output = AudioOutput::new(samples, 24000);

        // 40ms chunks at 24kHz = 960 samples per chunk
        let chunks = output.chunk(40);
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].len(), 960);
    }
}
