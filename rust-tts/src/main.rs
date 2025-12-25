//! Chatterbox TTS - Rust Inference Server
//!
//! High-performance Text-to-Speech inference service using Chatterbox models.
//! This server provides HTTP and WebSocket APIs for TTS generation with
//! support for voice cloning, expressiveness control, and streaming audio output.

use chatterbox_tts::{Config, Server};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chatterbox_tts=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load .env file if present
    if let Err(e) = dotenvy::dotenv() {
        tracing::debug!("No .env file found: {}", e);
    }

    // Load configuration from environment
    let config = Config::from_env();

    tracing::info!("Chatterbox TTS Server v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Configuration:");
    tracing::info!("  Host: {}:{}", config.server.host, config.server.port);
    tracing::info!("  Device: {}", config.model.device);
    tracing::info!("  Load Standard: {}", config.model.load_standard);
    tracing::info!("  Load Turbo: {}", config.model.load_turbo);
    tracing::info!("  Load Multilingual: {}", config.model.load_multilingual);
    tracing::info!("  Voices Directory: {}", config.voice.voices_directory.display());

    // Initialize Python
    let chatterbox_src = std::env::var("CHATTERBOX_SRC")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            // Try to find the src directory relative to the current directory
            let candidates = [
                PathBuf::from("../src"),
                PathBuf::from("./src"),
                PathBuf::from("/app/src"),
            ];
            candidates.into_iter().find(|p| p.join("chatterbox").exists())
        });

    tracing::info!("Initializing Python runtime...");
    chatterbox_tts::python::initialize_python(chatterbox_src.as_deref())?;

    // Get PyTorch info
    let (torch_version, device) = chatterbox_tts::python::get_torch_info();
    tracing::info!("PyTorch version: {}", torch_version);
    tracing::info!("Detected device: {}", device);

    // Create and configure server
    let server = Server::new(config);

    // Load models
    tracing::info!("Loading TTS models...");
    let load_start = std::time::Instant::now();
    server.load_models()?;
    tracing::info!("Models loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // Run server
    tracing::info!("Starting HTTP/WebSocket server...");
    server.run().await?;

    Ok(())
}
