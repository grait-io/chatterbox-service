//! Python interop utilities

use crate::error::{Result, TtsError};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

/// Global Python initialization state
static PYTHON_INITIALIZED: OnceCell<bool> = OnceCell::new();

/// Initialize Python runtime and set up the Chatterbox module path
pub fn initialize_python(chatterbox_src_path: Option<&Path>) -> Result<()> {
    PYTHON_INITIALIZED.get_or_init(|| {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Add the chatterbox source to Python path if provided
            if let Some(src_path) = chatterbox_src_path {
                let sys = py.import("sys").expect("Failed to import sys");
                let path = sys.getattr("path").expect("Failed to get sys.path");
                path.call_method1("insert", (0, src_path.to_str().unwrap_or("")))
                    .expect("Failed to insert path");
            }

            // Import and verify chatterbox is available
            match py.import("chatterbox") {
                Ok(_) => {
                    tracing::info!("Python Chatterbox module loaded successfully");
                }
                Err(e) => {
                    tracing::warn!("Failed to import chatterbox: {}", e);
                }
            }
        });

        true
    });

    Ok(())
}

/// Detect the best available device (cuda, mps, or cpu)
pub fn detect_device() -> String {
    Python::with_gil(|py| {
        let torch = match py.import("torch") {
            Ok(t) => t,
            Err(_) => return "cpu".to_string(),
        };

        // Check for CUDA
        if let Ok(cuda_available) = torch.getattr("cuda") {
            if let Ok(available) = cuda_available.call_method0("is_available") {
                if available.extract::<bool>().unwrap_or(false) {
                    return "cuda".to_string();
                }
            }
        }

        // Check for MPS (Apple Silicon)
        if let Ok(backends) = torch.getattr("backends") {
            if let Ok(mps) = backends.getattr("mps") {
                if let Ok(available) = mps.call_method0("is_available") {
                    if available.extract::<bool>().unwrap_or(false) {
                        return "mps".to_string();
                    }
                }
            }
        }

        "cpu".to_string()
    })
}

/// Get PyTorch version info
pub fn get_torch_info() -> (String, String) {
    Python::with_gil(|py| {
        let torch = match py.import("torch") {
            Ok(t) => t,
            Err(_) => return ("unknown".to_string(), "cpu".to_string()),
        };

        let version = torch
            .getattr("__version__")
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let device = detect_device();

        (version, device)
    })
}

/// Audio generation result from Python
#[derive(Debug, Clone)]
pub struct PyAudioResult {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

/// Generate TTS audio using Python directly
pub fn generate_tts(
    text: &str,
    audio_prompt_path: Option<&str>,
    model_type: &str,
    exaggeration: f32,
    cfg_weight: f32,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    language_id: Option<&str>,
) -> Result<PyAudioResult> {
    Python::with_gil(|py| {
        // Import the appropriate module
        let tts_module = match model_type {
            "turbo" => py.import("chatterbox.tts_turbo")?,
            "multilingual" => py.import("chatterbox.mtl_tts")?,
            _ => py.import("chatterbox.tts")?,
        };

        // Get the TTS class
        let tts_class = match model_type {
            "turbo" => tts_module.getattr("ChatterboxTurboTTS")?,
            "multilingual" => tts_module.getattr("ChatterboxMultilingualTTS")?,
            _ => tts_module.getattr("ChatterboxTTS")?,
        };

        // Load the model
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", detect_device())?;
        let model = tts_class.call_method("from_pretrained", (), Some(&kwargs))?;

        // Prepare generation kwargs
        let gen_kwargs = PyDict::new(py);
        gen_kwargs.set_item("text", text)?;

        if let Some(audio_path) = audio_prompt_path {
            gen_kwargs.set_item("audio_prompt_path", audio_path)?;
        }

        match model_type {
            "turbo" => {
                // Turbo doesn't support exaggeration/cfg_weight
            }
            _ => {
                gen_kwargs.set_item("exaggeration", exaggeration)?;
                gen_kwargs.set_item("cfg_weight", cfg_weight)?;
            }
        }

        if let Some(lang) = language_id {
            gen_kwargs.set_item("language_id", lang)?;
        }

        gen_kwargs.set_item("temperature", temperature)?;
        gen_kwargs.set_item("top_p", top_p)?;
        gen_kwargs.set_item("repetition_penalty", repetition_penalty)?;

        // Generate audio
        let result = model.call_method("generate", (), Some(&gen_kwargs))?;

        // Extract samples
        let wav = result.getattr("wav")?;
        let sr = result.getattr("sr")?;

        let wav_flat = wav.call_method0("flatten")?;
        let wav_list = wav_flat.call_method0("tolist")?;

        let samples: Vec<f32> = wav_list
            .iter()?
            .map(|x| x.and_then(|v| v.extract::<f32>()).unwrap_or(0.0))
            .collect();

        let sample_rate = sr.extract::<u32>().unwrap_or(24000);

        Ok(PyAudioResult {
            samples,
            sample_rate,
        })
    })
    .map_err(|e: PyErr| TtsError::Python(e.to_string()))
}

/// Load a voice reference and return speaker embeddings
pub fn load_voice_reference(audio_path: &str) -> Result<Vec<f32>> {
    Python::with_gil(|py| {
        let librosa = py.import("librosa")?;

        // Load audio at 16kHz
        let kwargs = PyDict::new(py);
        kwargs.set_item("sr", 16000)?;

        let result = librosa.call_method("load", (audio_path,), Some(&kwargs))?;
        let wav = result.get_item(0)?;

        // Convert to list
        let wav_list = wav.call_method0("tolist")?;
        let samples: Vec<f32> = wav_list
            .iter()?
            .map(|x| x.and_then(|v| v.extract::<f32>()).unwrap_or(0.0))
            .collect();

        Ok(samples)
    })
    .map_err(|e: PyErr| TtsError::Python(e.to_string()))
}

/// Check if a file is a valid audio file
pub fn is_valid_audio_file(path: &str) -> bool {
    Python::with_gil(|py| {
        let librosa = match py.import("librosa") {
            Ok(l) => l,
            Err(_) => return false,
        };

        let kwargs = PyDict::new(py);
        kwargs.set_item("sr", 16000).ok()?;

        librosa.call_method("load", (path,), Some(&kwargs)).is_ok()
    })
    .unwrap_or(false)
}

/// Get audio file duration in seconds
pub fn get_audio_duration(path: &str) -> Result<f32> {
    Python::with_gil(|py| {
        let librosa = py.import("librosa")?;

        let duration = librosa.call_method1("get_duration", (path,))?;
        let duration_secs = duration.extract::<f32>()?;

        Ok(duration_secs)
    })
    .map_err(|e: PyErr| TtsError::Python(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = detect_device();
        assert!(
            device == "cuda" || device == "mps" || device == "cpu",
            "Device should be cuda, mps, or cpu"
        );
    }
}
