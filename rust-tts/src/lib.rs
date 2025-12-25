//! # Chatterbox TTS - Rust Inference Service
//!
//! High-performance Text-to-Speech inference service using Chatterbox models.
//! This library provides a Rust interface to the Chatterbox TTS models with
//! support for voice cloning, expressiveness control, and streaming audio output.

pub mod config;
pub mod error;
pub mod models;
pub mod audio;
pub mod server;
pub mod python;

pub use config::Config;
pub use error::{TtsError, Result};
pub use models::{TtsModel, VoiceConfig, GenerationParams, AudioOutput};
pub use server::Server;
