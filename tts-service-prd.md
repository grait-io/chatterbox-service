
A small WebSocket‑based wrapper around `chatterbox-service` running in Docker with a single “send text, receive streamed audio” API is enough for your voice commenter use case.[^1][^2]

## Overview

- **Goal:** Self‑hosted TTS over WebSocket, deployable via Coolify (Docker Compose).[^1]
- **Core:** `grait-io/chatterbox-service` container for TTS, plus a lightweight gateway exposing a WebSocket streaming API.[^1]
- **Client:** Connect via WebSocket, send simple JSON messages with text, receive binary audio frames suitable for immediate playback.[^2]

***

## `prd.md` – Chatterbox TTS WebSocket Service

### 1. Product summary

- **Name:** Chatterbox TTS WebSocket Service
- **Purpose:** Low‑latency text‑to‑speech for “voice commenter” and similar real‑time commentary tools.
- **Deployment:** Coolify‑managed Docker Compose stack.
- **Primary UX:**
    - Client sends short text comments over WebSocket.
    - Service streams back audio frames (Opus or PCM) as they are generated.[^3]


### 2. Components \& deployment

- **Services (Docker Compose):**
    - `chatterbox-core`: runs `grait-io/chatterbox-service` (TTS engine).[^1]
    - `tts-gateway`: small Node/Go/Python WebSocket server that:
        - Accepts client WebSocket connections.
        - Proxies synth requests to `chatterbox-core`.
        - Streams audio frames back to client.[^2]
- **Runtime requirements:**
    - GPU (optional but recommended) for low latency TTS, configured via `NVIDIA_VISIBLE_DEVICES` etc.[^1]
    - Exposed public port: `8080` (HTTP for health) and `8081` (WebSocket).
- **Coolify integration:**
    - Stack defined in `docker-compose.yml` in repo root.
    - Coolify uses GitHub integration to build and run the stack on push to `main`.


### 3. WebSocket API design

#### 3.1 Connection

- **URL:** `wss://<host>/ws/tts` (or `ws://localhost:8081/ws/tts` in dev).
- **Protocol:**
    - Text frames: JSON control messages (requests, metadata, errors).
    - Binary frames: raw audio chunks (Opus or PCM).[^3][^2]
- **Authentication (MVP):**
    - Optional `?token=<API_TOKEN>` query param or `Authorization: Bearer <token>` header; can be disabled in dev.


#### 3.2 Message types (JSON)

All JSON messages use a **top‑level `type` field**:

- `ready` – sent by server on connection.
- `say` – sent by client to request TTS.
- `chunk` – optional text/progress events (JSON) before or during audio.
- `done` – sent by server when an utterance is finished.
- `error` – sent by server on errors.


##### 3.2.1 Server → client: `ready`

```json
{
  "type": "ready",
  "version": "1.0.0",
  "models": ["chatterbox-turbo", "chatterbox-multilingual"],
  "default_format": "opus"
}
```

- Indicates the connection is usable and describes **default codec**.[^1]


##### 3.2.2 Client → server: `say`

```json
{
  "type": "say",
  "id": "comment-123",
  "text": "That was an insane headshot!",
  "language_id": "en",
  "voice": {
    "audio_prompt_path": null
  },
  "audio": {
    "format": "opus",          // "opus" | "pcm_s16le"
    "sample_rate": 24000,      // ignored if format is opus
    "stream": true
  }
}
```

- `id`: client‑provided correlation id.
- `text`: required; the comment text.
- `language_id`: passed to the multilingual model when used (`"en"`, `"pl"`, `"fr"`, etc.).[^1]
- `voice.audio_prompt_path`: path/id for voice cloning (optional; server‑side configured).[^1]
- `audio.format`: desired output format.
- `audio.stream`: must be `true` for streaming; `false` would request a single aggregated response (future).


##### 3.2.3 Optional: Server → client: `chunk` (JSON)

```json
{
  "type": "chunk",
  "id": "comment-123",
  "state": "speaking",
  "offset_ms": 0
}
```

- Used if you want progress or incremental text alignment; can be omitted in MVP.


##### 3.2.4 Server → client: binary audio frames

- Binary WebSocket frames, with **no JSON wrapping**, each representing **20–40 ms** of audio:[^3]
    - If `format="opus"`: each frame is an Opus packet in an agreed configuration (e.g. mono, 24 kHz).
    - If `format="pcm_s16le"`: raw little‑endian 16‑bit PCM, mono, at requested sample rate.
- Frames for a given `say` request are contiguous; server ensures no interleaving between ids.


##### 3.2.5 Server → client: `done`

```json
{
  "type": "done",
  "id": "comment-123",
  "duration_ms": 1850
}
```

- Sent after the last audio frame for that `id`.
- Allows client to stop playback or flush remaining buffers.


##### 3.2.6 Server → client: `error`

```json
{
  "type": "error",
  "id": "comment-123",
  "code": "SYNTH_FAILED",
  "message": "Chatterbox generation failed"
}
```

- Non‑fatal errors keep the WebSocket open if possible; fatal errors may close it.


### 4. Server behavior

- On connection:
    - Validate token (if enabled).
    - Send `ready`.
- On `say`:
    - Validate payload (text length limits, language support).
    - Select model:
        - `chatterbox-turbo` for English default.
        - `chatterbox-multilingual` when `language_id != "en"`.[^1]
    - Call `generate` in streaming mode (looping synthesis) and:
        - Immediately push each generated chunk as a binary frame.
        - After synthesis completes, send `done`.[^3]
- Backpressure / concurrency:
    - Per‑client limit (e.g. max 3 concurrent active `say` ids).
    - Optional queueing strategy (for voice commenter, simple FIFO is enough).


### 5. Docker Compose (MVP sketch)

```yaml
version: "3.9"

services:
  chatterbox-core:
    image: ghcr.io/grait-io/chatterbox-service:latest
    environment:
      DEVICE: "cuda"          # or "cpu"
      MODEL_VERSION: "turbo"  # or envs used by upstream service
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    volumes:
      - ./voices:/app/voices
    networks:
      - tts-net

  tts-gateway:
    build: ./gateway   # small WebSocket server
    environment:
      TTS_HOST: "chatterbox-core"
      TTS_PORT: "8000"
      AUDIO_FORMAT_DEFAULT: "opus"
      API_TOKEN: "${API_TOKEN:-}"
    ports:
      - "8081:8081"
    depends_on:
      - chatterbox-core
    networks:
      - tts-net

networks:
  tts-net:
    driver: bridge
```

- `gateway` talks to `chatterbox-core` over the internal network using the Python APIs or HTTP endpoints exposed by `chatterbox-service`.[^1]

***

## Client API usage (example)

### JavaScript browser client (Opus)

```js
const ws = new WebSocket("wss://your-tts.example.com/ws/tts");

ws.binaryType = "arraybuffer";

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "say",
    id: "comment-123",
    text: "Ok, next pull request looks promising.",
    language_id: "en",
    audio: { format: "opus", stream: true }
  }));
};

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "done") {
      // stop enqueueing
    }
    if (msg.type === "error") {
      console.error(msg);
    }
    return;
  }

  const audioData = new Uint8Array(event.data);
  // enqueue into Opus decoder / Web Audio playback buffer
};
```

- Client decodes Opus frames with a small WASM decoder (e.g. `opusscript`/custom) and feeds Web Audio buffers for playback.[^3]

***

If you want, a concrete `gateway` implementation (e.g. FastAPI + `websockets`, or Node + `ws`) can be sketched next using this `prd.md` as the contract.

<div align="center">⁂</div>

[^1]: https://github.com/grait-io/chatterbox-service

[^2]: https://platform.openai.com/docs/guides/realtime

[^3]: https://picovoice.ai/blog/streaming-text-to-speech-for-ai-agents/

