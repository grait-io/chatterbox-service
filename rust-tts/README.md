# Chatterbox TTS - Rust Inference Service

High-performance Text-to-Speech inference service using Chatterbox models, implemented in Rust with Python interop via PyO3.

## Features

- **High Performance**: Rust-based server with async I/O using Tokio and Axum
- **Model Preloading**: Models are loaded on startup for instant inference
- **Voice Cloning**: Set a reference voice file for all TTS requests
- **Multiple Models**: Support for Standard (expressive), Turbo (fast), and Multilingual models
- **Streaming Audio**: WebSocket API with chunked audio streaming
- **HTTP API**: RESTful endpoints for synchronous TTS generation
- **Flexible Configuration**: Environment-based configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust TTS Server                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   HTTP API   │  │ WebSocket API│  │  Voice Manager   │  │
│  │  (Axum)      │  │  (Axum WS)   │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────▼─────────────────▼────────────────────▼─────────┐  │
│  │                  TTS Model Manager                     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │  Standard   │ │   Turbo     │ │  Multilingual   │  │  │
│  │  │  (PyO3)     │ │   (PyO3)    │ │    (PyO3)       │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Python Runtime (PyO3)                     │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │           Chatterbox TTS Models                 │  │  │
│  │  │  - T3 (Text-to-Token)                          │  │  │
│  │  │  - S3Gen (Token-to-Waveform)                   │  │  │
│  │  │  - Voice Encoder                               │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Docker

```bash
# CPU version
docker-compose -f docker-compose.yml up -d

# GPU version
docker-compose -f docker-compose.gpu.yml up -d
```

### Building from Source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cd rust-tts
cargo build --release

# Run
./target/release/chatterbox-server
```

## API Reference

### HTTP Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "loaded_models": ["standard"],
  "device": "cuda",
  "version": "0.1.0"
}
```

#### Generate TTS
```http
POST /api/tts
Content-Type: application/json

{
  "text": "Hello, world!",
  "model": "standard",
  "voice": {
    "audio_prompt_path": "/app/voices/my_voice.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  },
  "audio": {
    "format": "wav"
  }
}
```

Response: Audio file (WAV or PCM)

#### Set Default Voice
```http
POST /api/voice/default
Content-Type: application/json

{
  "voice_path": "my_voice.wav"
}
```

#### Get Default Voice
```http
GET /api/voice/default
```

#### List Available Voices
```http
GET /api/voices
```

#### Upload Voice File
```http
POST /api/voices/upload
Content-Type: multipart/form-data

file: <audio file>
```

### WebSocket API

Connect to `/ws/tts` for real-time streaming TTS.

#### Ready Message (Server → Client)
```json
{
  "type": "ready",
  "models": ["standard"],
  "device": "cuda"
}
```

#### Say Request (Client → Server)
```json
{
  "type": "say",
  "id": "unique-request-id",
  "text": "Hello, world!",
  "model": "standard",
  "language_id": "en",
  "voice": {
    "audio_prompt_path": "/app/voices/my_voice.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  },
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 24000,
    "stream": true,
    "chunk_size_ms": 40
  }
}
```

#### Chunk Message (Server → Client)
```json
{
  "type": "chunk",
  "id": "unique-request-id",
  "index": 0,
  "data": "<base64-encoded-audio>",
  "is_last": false
}
```

#### Done Message (Server → Client)
```json
{
  "type": "done",
  "id": "unique-request-id",
  "duration_seconds": 2.5
}
```

#### Set Voice Request (Client → Server)
```json
{
  "type": "set_voice",
  "voice_path": "my_voice.wav"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8081` | Server port |
| `DEVICE` | `auto` | Device to use (`auto`, `cpu`, `cuda`, `mps`) |
| `LOAD_STANDARD` | `true` | Load standard (expressive) model |
| `LOAD_TURBO` | `false` | Load turbo (fast) model |
| `LOAD_MULTILINGUAL` | `false` | Load multilingual model |
| `MAX_CONCURRENT_REQUESTS` | `3` | Maximum concurrent TTS requests |
| `MAX_TEXT_LENGTH` | `5000` | Maximum text length in characters |
| `VOICES_DIR` | `./voices` | Directory for voice reference files |
| `DEFAULT_VOICE` | | Default voice file path |
| `HF_HOME` | | HuggingFace cache directory |
| `API_TOKEN` | | API authentication token (optional) |
| `RUST_LOG` | `info` | Logging level |

## Voice Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `exaggeration` | 0.0-2.0 | 0.5 | Emotional intensity (standard only) |
| `cfg_weight` | 0.0-1.0 | 0.5 | Pacing control (standard only) |
| `temperature` | 0.1-1.5 | 0.8 | Sampling diversity |
| `top_p` | 0.0-1.0 | 0.95 | Nucleus sampling threshold |
| `repetition_penalty` | ≥1.0 | 1.2 | Repetition penalty |

## Voice Reference Files

- **Format**: WAV, MP3, FLAC
- **Duration**: 5-10 seconds of clear speech
- **Quality**: 16kHz+ sample rate recommended
- **Content**: Clear speech without background noise

### Setting Default Voice

You can set a default voice that will be used for all TTS requests:

```bash
# Via HTTP API
curl -X POST http://localhost:8081/api/voice/default \
  -H "Content-Type: application/json" \
  -d '{"voice_path": "my_voice.wav"}'

# Via WebSocket
{"type": "set_voice", "voice_path": "my_voice.wav"}
```

## Examples

### Python Client

```python
import requests
import json

# Generate TTS
response = requests.post(
    "http://localhost:8081/api/tts",
    json={
        "text": "Hello, this is a test.",
        "voice": {
            "exaggeration": 0.7,
            "cfg_weight": 0.5
        }
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### WebSocket Client (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8081/ws/tts");

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "ready") {
        // Server is ready, send request
        ws.send(JSON.stringify({
            type: "say",
            id: "request-1",
            text: "Hello, world!",
            voice: {
                exaggeration: 0.5
            },
            audio: {
                stream: true,
                chunk_size_ms: 40
            }
        }));
    } else if (msg.type === "chunk") {
        // Process audio chunk
        const audioData = atob(msg.data);
        // Play or buffer audio...
    } else if (msg.type === "done") {
        console.log(`Generation complete: ${msg.duration_seconds}s`);
    }
};
```

## Performance

| Model | Device | Latency (first token) | RTF |
|-------|--------|----------------------|-----|
| Standard | CUDA (3090) | ~200ms | 0.15x |
| Standard | CPU (8 cores) | ~2s | 1.5x |
| Turbo | CUDA (3090) | ~100ms | 0.08x |
| Turbo | CPU (8 cores) | ~1s | 0.8x |

*RTF = Real-Time Factor (lower is better)*

## License

MIT License - see LICENSE file for details.
