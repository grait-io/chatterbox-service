# Chatterbox TTS

A high-performance Rust library for Text-to-Speech synthesis using [Chatterbox](https://github.com/resemble-ai/chatterbox) models.

[![Crates.io](https://img.shields.io/crates/v/chatterbox-tts.svg)](https://crates.io/crates/chatterbox-tts)
[![Documentation](https://docs.rs/chatterbox-tts/badge.svg)](https://docs.rs/chatterbox-tts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Voice Cloning**: Clone any voice from a short audio sample (5-10 seconds)
- **Expressive Speech**: Control emotional intensity and pacing
- **Multiple Models**: Standard (expressive), Turbo (fast), Multilingual (23 languages)
- **Streaming**: Real-time audio streaming for low-latency applications
- **Easy Integration**: Simple API for embedding TTS in your Rust applications
- **Optional Server**: HTTP/WebSocket server available via feature flag

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
chatterbox-tts = "0.1"
```

For server functionality:

```toml
[dependencies]
chatterbox-tts = { version = "0.1", features = ["server"] }
```

### Prerequisites

- Python 3.10+ with the Chatterbox package installed
- PyTorch with CUDA support (optional, for GPU acceleration)

## Quick Start

### Basic Usage

```rust
use chatterbox_tts::ChatterboxTts;

fn main() -> chatterbox_tts::Result<()> {
    // Initialize the TTS engine
    let mut tts = ChatterboxTts::new()?;

    // Load the model (downloads on first run)
    tts.load_model()?;

    // Generate speech
    let audio = tts.synthesize("Hello, world!")?;

    // Save to file
    audio.save_wav("output.wav")?;

    Ok(())
}
```

### Voice Cloning

```rust
use chatterbox_tts::{ChatterboxTts, VoiceConfig};

fn main() -> chatterbox_tts::Result<()> {
    let mut tts = ChatterboxTts::new()?;
    tts.load_model()?;

    // Set a reference voice (5-10 seconds of clear speech)
    tts.set_voice("path/to/reference.wav")?;

    // Generate with custom parameters
    let audio = tts.synthesize_with_config(
        "This will sound like the reference voice!",
        VoiceConfig::default()
            .exaggeration(0.7)  // Emotional intensity
            .cfg_weight(0.5),   // Pacing control
    )?;

    audio.save_wav("cloned_voice.wav")?;
    Ok(())
}
```

### Using the Builder Pattern

```rust
use chatterbox_tts::{ChatterboxTtsBuilder, ModelType, VoiceConfig};

fn main() -> chatterbox_tts::Result<()> {
    let tts = ChatterboxTtsBuilder::new()
        .device("cuda")                    // Use GPU
        .model_type(ModelType::Standard)   // Expressive model
        .voice("path/to/voice.wav")        // Default voice
        .default_config(
            VoiceConfig::default()
                .exaggeration(0.6)
                .temperature(0.9)
        )
        .auto_load()                       // Load model immediately
        .with_warmup()                     // Warmup for fast inference
        .build()?;

    let audio = tts.synthesize("Ready for fast inference!")?;
    audio.save_wav("output.wav")?;

    Ok(())
}
```

### Streaming Audio

```rust
use chatterbox_tts::ChatterboxTts;

fn main() -> chatterbox_tts::Result<()> {
    let mut tts = ChatterboxTts::new()?;
    tts.load_model()?;

    let audio = tts.synthesize("Long text for streaming...")?;

    // Chunk into 40ms pieces for real-time playback
    let chunks = audio.chunk_pcm(40);

    for chunk in chunks {
        // Send chunk to audio player or network stream
        play_audio_chunk(&chunk);
    }

    Ok(())
}
```

### Using the Prelude

```rust
use chatterbox_tts::prelude::*;

fn main() -> Result<()> {
    let mut tts = ChatterboxTts::new()?;
    tts.load_model()?;

    let audio = tts.synthesize("Hello!")?;
    audio.save_wav("hello.wav")?;

    Ok(())
}
```

## API Reference

### ChatterboxTts

The main TTS interface.

```rust
// Creation
ChatterboxTts::new()?;                    // Default settings
ChatterboxTts::with_device("cuda")?;      // Specific device
ChatterboxTts::with_config(config)?;      // Custom configuration

// Model loading
tts.load_model()?;                        // Load standard model
tts.load_model_type(ModelType::Turbo)?;   // Load specific model
tts.load_all_models()?;                   // Load all models
tts.warmup()?;                            // Warmup for fast inference

// Voice configuration
tts.set_voice("path/to/voice.wav")?;      // Set default voice
tts.clear_voice();                        // Clear default voice
tts.set_default_config(config);           // Set default parameters

// Synthesis
tts.synthesize("text")?;                  // Using defaults
tts.synthesize_with_config("text", config)?;  // Custom parameters
tts.synthesize_multilingual("text", "es")?;   // Multilingual
```

### VoiceConfig

Voice and generation parameters.

```rust
VoiceConfig::default()
    .exaggeration(0.7)        // 0.0-2.0, emotional intensity
    .cfg_weight(0.5)          // 0.0-1.0, pacing control
    .temperature(0.8)         // 0.1-1.5, sampling diversity
    .top_p(0.95)              // 0.0-1.0, nucleus sampling
    .repetition_penalty(1.2)  // ≥1.0, repetition penalty
```

### AudioOutput

Generated audio with utility methods.

```rust
audio.save_wav("output.wav")?;      // Save as WAV
audio.save_pcm("output.pcm")?;      // Save as raw PCM
audio.to_wav();                     // Get WAV bytes
audio.to_pcm_s16le();               // Get PCM bytes
audio.chunk(40);                    // Chunk into 40ms pieces
audio.chunk_pcm(40);                // Chunk as PCM bytes
audio.duration_seconds;             // Audio duration
audio.sample_rate;                  // Sample rate (24000)
```

## Examples

Run the examples:

```bash
# Simple TTS
cargo run --example simple_tts

# Voice cloning
cargo run --example voice_cloning -- path/to/voice.wav

# Streaming (requires server feature)
cargo run --example streaming --features server
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Core TTS functionality only |
| `server` | HTTP/WebSocket server |
| `full` | All features enabled |

## Server Mode

When built with the `server` feature, you can run a standalone TTS server:

```bash
cargo run --bin chatterbox-server --features server
```

Or use Docker:

```bash
# CPU version
docker-compose -f docker-compose.yml up -d

# GPU version
docker-compose -f docker-compose.gpu.yml up -d
```

### Server API

#### HTTP Endpoints

```http
POST /api/tts
Content-Type: application/json

{
  "text": "Hello, world!",
  "voice": {
    "exaggeration": 0.5,
    "cfg_weight": 0.5
  }
}
```

#### WebSocket Streaming

```javascript
const ws = new WebSocket("ws://localhost:8081/ws/tts");

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "chunk") {
        // Process audio chunk
    }
};

ws.send(JSON.stringify({
    type: "say",
    text: "Hello!",
    audio: { stream: true, chunk_size_ms: 40 }
}));
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `CHATTERBOX_SRC` | | Path to Chatterbox Python source |
| `HF_HOME` | | HuggingFace cache directory |

### Server-specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8081` | Server port |
| `LOAD_STANDARD` | `true` | Load standard model |
| `LOAD_TURBO` | `false` | Load turbo model |
| `VOICES_DIR` | `./voices` | Voice files directory |

## Voice Parameters

| Parameter | Range | Default | Model | Description |
|-----------|-------|---------|-------|-------------|
| `exaggeration` | 0.0-2.0 | 0.5 | Standard | Emotional intensity |
| `cfg_weight` | 0.0-1.0 | 0.5 | Standard | Pacing control |
| `temperature` | 0.1-1.5 | 0.8 | All | Sampling diversity |
| `top_p` | 0.0-1.0 | 0.95 | All | Nucleus sampling |
| `repetition_penalty` | ≥1.0 | 1.2 | All | Repetition penalty |

## Supported Languages (Multilingual Model)

English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian,
Swedish, Norwegian, Danish, Finnish, Greek, Chinese, Japanese, Korean, Hindi,
Arabic, Hebrew, Turkish, Malay, Swahili

## Performance

| Model | Device | First Token | RTF |
|-------|--------|-------------|-----|
| Standard | CUDA (3090) | ~200ms | 0.15x |
| Standard | CPU (8 cores) | ~2s | 1.5x |
| Turbo | CUDA (3090) | ~100ms | 0.08x |
| Turbo | CPU (8 cores) | ~1s | 0.8x |

*RTF = Real-Time Factor (lower is better)*

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
