//! Audio processing utilities

use crate::error::{Result, TtsError};
use std::io::Cursor;
use std::path::Path;

/// Sample rate used by Chatterbox models
pub const MODEL_SAMPLE_RATE: u32 = 24000;

/// Sample rate for voice encoding
pub const VOICE_ENCODER_SAMPLE_RATE: u32 = 16000;

/// Audio format for output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// Raw PCM signed 16-bit little-endian
    PcmS16Le,
    /// WAV format
    Wav,
    /// Opus encoded (for streaming)
    Opus,
}

impl std::str::FromStr for AudioFormat {
    type Err = TtsError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pcm_s16le" | "pcm" => Ok(AudioFormat::PcmS16Le),
            "wav" => Ok(AudioFormat::Wav),
            "opus" => Ok(AudioFormat::Opus),
            _ => Err(TtsError::InvalidParameter(format!(
                "Unknown audio format: {}",
                s
            ))),
        }
    }
}

/// Audio chunk for streaming
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Chunk index (0-based)
    pub index: usize,
    /// Audio data in the specified format
    pub data: Vec<u8>,
    /// Whether this is the last chunk
    pub is_last: bool,
    /// Timestamp in milliseconds from start
    pub timestamp_ms: u64,
}

/// Audio buffer for accumulating and chunking audio
pub struct AudioBuffer {
    /// Raw f32 samples at model sample rate
    samples: Vec<f32>,
    /// Output format
    format: AudioFormat,
    /// Chunk size in samples
    chunk_size_samples: usize,
    /// Current chunk index
    current_chunk: usize,
    /// Sample rate
    sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(format: AudioFormat, chunk_size_ms: u32, sample_rate: u32) -> Self {
        let chunk_size_samples = (sample_rate * chunk_size_ms / 1000) as usize;
        Self {
            samples: Vec::new(),
            format,
            chunk_size_samples,
            current_chunk: 0,
            sample_rate,
        }
    }

    /// Add samples to the buffer
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    /// Get available chunks
    pub fn drain_chunks(&mut self) -> Vec<AudioChunk> {
        let mut chunks = Vec::new();

        while self.samples.len() >= self.chunk_size_samples {
            let chunk_samples: Vec<f32> = self.samples.drain(..self.chunk_size_samples).collect();
            let data = self.encode_chunk(&chunk_samples);

            let timestamp_ms =
                (self.current_chunk * self.chunk_size_samples * 1000 / self.sample_rate as usize)
                    as u64;

            chunks.push(AudioChunk {
                index: self.current_chunk,
                data,
                is_last: false,
                timestamp_ms,
            });

            self.current_chunk += 1;
        }

        chunks
    }

    /// Finalize and get remaining samples as the last chunk
    pub fn finalize(mut self) -> Option<AudioChunk> {
        if self.samples.is_empty() {
            return None;
        }

        let data = self.encode_chunk(&self.samples);
        let timestamp_ms =
            (self.current_chunk * self.chunk_size_samples * 1000 / self.sample_rate as usize)
                as u64;

        Some(AudioChunk {
            index: self.current_chunk,
            data,
            is_last: true,
            timestamp_ms,
        })
    }

    /// Encode samples to the output format
    fn encode_chunk(&self, samples: &[f32]) -> Vec<u8> {
        match self.format {
            AudioFormat::PcmS16Le => samples_to_pcm_s16le(samples),
            AudioFormat::Wav => samples_to_wav(samples, self.sample_rate),
            AudioFormat::Opus => {
                // For now, fall back to PCM
                // TODO: Add Opus encoding
                samples_to_pcm_s16le(samples)
            }
        }
    }
}

/// Convert f32 samples to PCM S16LE bytes
pub fn samples_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&int_sample.to_le_bytes());
    }
    bytes
}

/// Convert PCM S16LE bytes to f32 samples
pub fn pcm_s16le_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32767.0
        })
        .collect()
}

/// Convert f32 samples to WAV bytes
pub fn samples_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let int_sample = (clamped * 32767.0) as i16;
            writer.write_sample(int_sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    cursor.into_inner()
}

/// Load audio file and return f32 samples at target sample rate
pub fn load_audio_file<P: AsRef<Path>>(path: P, target_sample_rate: u32) -> Result<Vec<f32>> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(TtsError::VoiceFileNotFound(path.display().to_string()));
    }

    let reader = hound::WavReader::open(path)
        .map_err(|e| TtsError::AudioProcessing(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .into_samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / 32767.0)
                .collect(),
            24 => reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / 8388607.0)
                .collect(),
            32 => reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / 2147483647.0)
                .collect(),
            _ => {
                return Err(TtsError::AudioProcessing(format!(
                    "Unsupported bit depth: {}",
                    spec.bits_per_sample
                )))
            }
        },
    };

    // Convert to mono if stereo
    let mono_samples = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        samples
    };

    // Resample if necessary
    if spec.sample_rate != target_sample_rate {
        resample(&mono_samples, spec.sample_rate, target_sample_rate)
    } else {
        Ok(mono_samples)
    }
}

/// Resample audio from one sample rate to another
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        samples.len(),
        1, // mono
    )
    .map_err(|e| TtsError::AudioProcessing(format!("Resampler creation failed: {}", e)))?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| TtsError::AudioProcessing(format!("Resampling failed: {}", e)))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}

/// Normalize audio samples to -1.0 to 1.0 range
pub fn normalize_audio(samples: &mut [f32]) {
    let max_val = samples
        .iter()
        .map(|s| s.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);

    if max_val > 0.0 && max_val != 1.0 {
        for sample in samples.iter_mut() {
            *sample /= max_val;
        }
    }
}

/// Trim silence from the beginning and end of audio
pub fn trim_silence(samples: &[f32], threshold_db: f32) -> Vec<f32> {
    let threshold = 10f32.powf(threshold_db / 20.0);

    // Find start
    let start = samples
        .iter()
        .position(|&s| s.abs() > threshold)
        .unwrap_or(0);

    // Find end
    let end = samples
        .iter()
        .rposition(|&s| s.abs() > threshold)
        .unwrap_or(samples.len());

    if start < end {
        samples[start..=end].to_vec()
    } else {
        samples.to_vec()
    }
}

/// Calculate audio duration in seconds
pub fn duration_seconds(samples: &[f32], sample_rate: u32) -> f32 {
    samples.len() as f32 / sample_rate as f32
}

/// Calculate RMS (root mean square) of audio
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Calculate dB level from RMS
pub fn rms_to_db(rms_value: f32) -> f32 {
    if rms_value > 0.0 {
        20.0 * rms_value.log10()
    } else {
        f32::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_conversion() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let pcm = samples_to_pcm_s16le(&samples);
        let back = pcm_s16le_to_samples(&pcm);

        for (orig, conv) in samples.iter().zip(back.iter()) {
            assert!((orig - conv).abs() < 0.001);
        }
    }

    #[test]
    fn test_audio_buffer() {
        let mut buffer = AudioBuffer::new(AudioFormat::PcmS16Le, 10, 1000);

        // 10ms at 1kHz = 10 samples
        let samples: Vec<f32> = (0..25).map(|i| i as f32 / 25.0).collect();
        buffer.push_samples(&samples);

        let chunks = buffer.drain_chunks();
        assert_eq!(chunks.len(), 2); // 25 samples = 2 full chunks + 5 remaining
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[1].index, 1);

        let final_chunk = buffer.finalize();
        assert!(final_chunk.is_some());
        assert!(final_chunk.unwrap().is_last);
    }

    #[test]
    fn test_normalize() {
        let mut samples = vec![0.0, 0.25, -0.5];
        normalize_audio(&mut samples);
        assert!((samples[2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_duration() {
        let samples = vec![0.0; 24000];
        let duration = duration_seconds(&samples, 24000);
        assert!((duration - 1.0).abs() < 0.001);
    }
}
