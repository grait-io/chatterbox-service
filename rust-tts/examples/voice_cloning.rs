//! Voice Cloning Example
//!
//! This example demonstrates voice cloning using a reference audio file.
//! The synthesized speech will sound like the voice in the reference file.
//!
//! # Running
//!
//! ```bash
//! cargo run --example voice_cloning -- path/to/reference.wav
//! ```

use chatterbox_tts::{ChatterboxTts, ChatterboxTtsBuilder, ModelType, Result, VoiceConfig};
use std::env;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Get the voice reference file from command line args
    let args: Vec<String> = env::args().collect();
    let voice_path = args.get(1).map(|s| s.as_str());

    println!("=== Chatterbox Voice Cloning Example ===\n");

    // Method 1: Using the builder pattern
    println!("Method 1: Using ChatterboxTtsBuilder");
    println!("--------------------------------------");

    let mut builder = ChatterboxTtsBuilder::new()
        .device("auto")
        .model_type(ModelType::Standard)
        .default_config(
            VoiceConfig::default()
                .exaggeration(0.6)
                .cfg_weight(0.5)
                .temperature(0.8),
        )
        .auto_load()
        .with_warmup();

    // Add voice if provided
    if let Some(path) = voice_path {
        println!("Using voice reference: {}", path);
        builder = builder.voice(path);
    }

    let tts = builder.build()?;

    // Generate speech
    let text = "Hello! I'm speaking with a cloned voice. \
                This demonstrates the voice cloning capabilities of Chatterbox.";

    println!("Generating speech...");
    let audio = tts.synthesize(text)?;
    audio.save_wav("voice_clone_builder.wav")?;
    println!("Saved to: voice_clone_builder.wav ({:.2}s)\n", audio.duration_seconds);

    // Method 2: Using set_voice after creation
    println!("Method 2: Using set_voice method");
    println!("---------------------------------");

    let mut tts2 = ChatterboxTts::new()?;
    tts2.load_model()?;

    if let Some(path) = voice_path {
        tts2.set_voice(path)?;
        println!("Voice set to: {}", path);
    }

    // Generate with different expressiveness levels
    let configs = [
        ("low_expression.wav", 0.2, "Low expressiveness"),
        ("medium_expression.wav", 0.5, "Medium expressiveness"),
        ("high_expression.wav", 0.9, "High expressiveness"),
    ];

    for (filename, exaggeration, description) in configs {
        println!("Generating {} (exaggeration={})", description, exaggeration);

        let audio = tts2.synthesize_with_config(
            "This is a test with different expression levels.",
            VoiceConfig::default().exaggeration(exaggeration),
        )?;

        audio.save_wav(filename)?;
        println!("  Saved to: {} ({:.2}s)", filename, audio.duration_seconds);
    }

    println!("\n=== Done! ===");
    println!("Check the generated WAV files to hear the differences.");

    Ok(())
}
