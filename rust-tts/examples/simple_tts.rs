//! Simple TTS Example
//!
//! This example demonstrates basic text-to-speech synthesis using the
//! chatterbox-tts library.
//!
//! # Running
//!
//! ```bash
//! cargo run --example simple_tts
//! ```

use chatterbox_tts::{ChatterboxTts, Result};

fn main() -> Result<()> {
    // Initialize logging (optional)
    tracing_subscriber::fmt::init();

    println!("Initializing Chatterbox TTS...");

    // Create a new TTS instance
    let mut tts = ChatterboxTts::new()?;

    // Load the model (this may take a moment on first run)
    println!("Loading model...");
    tts.load_model()?;

    // Optional: warmup the model for faster first inference
    println!("Warming up model...");
    tts.warmup()?;

    // Synthesize speech
    println!("Synthesizing speech...");
    let audio = tts.synthesize("Hello! This is a test of the Chatterbox text to speech system.")?;

    // Print audio info
    println!("Generated audio:");
    println!("  Duration: {:.2}s", audio.duration_seconds);
    println!("  Sample rate: {} Hz", audio.sample_rate);
    println!("  Samples: {}", audio.samples.len());

    // Save to file
    let output_path = "output.wav";
    audio.save_wav(output_path)?;
    println!("Saved to: {}", output_path);

    Ok(())
}
