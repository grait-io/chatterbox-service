# LoL Commenter App Integration Guide

Build real-time League of Legends commentary using the Chatterbox TTS WebSocket API.

## Quick Start

```javascript
const ws = new WebSocket('ws://localhost:8081/ws/tts');
ws.binaryType = 'arraybuffer';

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'say',
    id: 'pentakill-001',
    text: 'PENTAKILL! FAKER DOES IT AGAIN!',
    model: 'standard',
    voice: { exaggeration: 1.4, cfg_weight: 0.2, temperature: 0.9 }
  }));
};

ws.onmessage = (e) => {
  if (e.data instanceof ArrayBuffer) {
    // PCM audio chunk - play it
    playAudio(e.data);
  } else {
    const msg = JSON.parse(e.data);
    if (msg.type === 'done') console.log('Finished:', msg.duration_ms + 'ms');
  }
};
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Connection & Authentication](#connection--authentication)
3. [Message Protocol](#message-protocol)
4. [Voice Parameters Deep Dive](#voice-parameters-deep-dive)
5. [Commentary Presets](#commentary-presets)
6. [Audio Playback](#audio-playback)
7. [Real-Time Commentary Patterns](#real-time-commentary-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Error Handling](#error-handling)
10. [Complete Example App](#complete-example-app)

---

## Architecture Overview

```
┌─────────────────────┐     WebSocket      ┌──────────────────────┐
│   Your LoL App      │◄──────────────────►│   TTS Gateway        │
│   (Commentary AI)   │    JSON + Binary   │   (Port 8081)        │
└─────────────────────┘                    └──────────────────────┘
         │                                           │
         │                                           ▼
         │                                  ┌──────────────────────┐
         ▼                                  │   Chatterbox Models  │
┌─────────────────────┐                     │   - Standard (expr.) │
│   Game Events       │                     │   - Turbo (fast)     │
│   (Riot API/OCR)    │                     └──────────────────────┘
└─────────────────────┘
```

**Flow:**
1. Your app detects game events (kills, objectives, plays)
2. Generate commentary text (LLM or templates)
3. Send to TTS with appropriate voice settings
4. Receive streamed audio and play in real-time

---

## Connection & Authentication

### Basic Connection

```javascript
const WS_URL = 'ws://localhost:8081/ws/tts';
// Production: 'wss://your-domain.com/ws/tts'

const ws = new WebSocket(WS_URL);
ws.binaryType = 'arraybuffer'; // Required for audio chunks
```

### With Authentication

```javascript
const token = 'your-api-token';
const ws = new WebSocket(`${WS_URL}?token=${token}`);
```

### Connection Lifecycle

```javascript
ws.onopen = () => {
  console.log('Connected, waiting for ready...');
};

ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const msg = JSON.parse(event.data);

    if (msg.type === 'ready') {
      console.log('Server ready:', msg.models);
      // Now safe to send requests
    }
  }
};

ws.onclose = (event) => {
  console.log('Disconnected:', event.code, event.reason);
  // Implement reconnection logic
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

### Ready Message

When connected, server sends:

```json
{
  "type": "ready",
  "version": "1.0.0",
  "models": ["chatterbox-standard", "chatterbox-turbo"],
  "default_format": "opus"
}
```

---

## Message Protocol

### Request: `say`

Send text for synthesis:

```json
{
  "type": "say",
  "id": "unique-request-id",
  "text": "PENTAKILL! The crowd goes wild!",
  "language_id": "en",
  "model": "standard",
  "voice": {
    "exaggeration": 1.2,
    "cfg_weight": 0.3,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "audio_prompt_path": null
  },
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 24000,
    "stream": true
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Always `"say"` |
| `id` | string | Yes | Your correlation ID for tracking |
| `text` | string | Yes | Commentary text (max 5000 chars) |
| `language_id` | string | No | `"en"` default |
| `model` | string | No | `"standard"` (expressive) or `"turbo"` (fast) |
| `voice` | object | No | Voice parameters (see below) |
| `audio` | object | No | Output format settings |

### Response: Binary Audio Chunks

Audio arrives as binary WebSocket frames:
- Format: PCM S16LE (16-bit signed little-endian)
- Sample rate: 24000 Hz
- Channels: Mono
- Chunk size: ~40ms (~1920 bytes per chunk)

### Response: `done`

Sent after last audio chunk:

```json
{
  "type": "done",
  "id": "unique-request-id",
  "duration_ms": 3500
}
```

### Response: `error`

```json
{
  "type": "error",
  "id": "unique-request-id",
  "code": "SYNTH_FAILED",
  "message": "Text too long"
}
```

Error codes:
- `INVALID_JSON` - Malformed request
- `INVALID_REQUEST` - Missing/invalid fields
- `TEXT_TOO_LONG` - Exceeds 5000 chars
- `TOO_MANY_REQUESTS` - Rate limited
- `SYNTH_FAILED` - Generation error

---

## Voice Parameters Deep Dive

### Model Selection

| Model | Speed | Expressiveness | Use Case |
|-------|-------|----------------|----------|
| `standard` | ~13 tokens/s | Full control | Hype moments, varied emotions |
| `turbo` | ~35 tokens/s | Fixed | Rapid updates, filler commentary |

**Rule of thumb:** Use `standard` for important moments, `turbo` for frequent updates.

### Exaggeration (0.0 - 2.0)

Controls emotional intensity. Think of it as the "hype dial":

```
0.0 ──────────────────────────────────────────────────── 2.0
│         │              │              │              │
Monotone  Analyst        Normal         Excited        MAXIMUM
          Desk           Play-by-play   Casting        OVERDRIVE
```

| Value | Mood | Example Moment |
|-------|------|----------------|
| 0.0-0.3 | Calm, measured | Post-game analysis, stats readout |
| 0.4-0.6 | Professional | Lane swap, ward placement |
| 0.7-0.9 | Engaged | Successful gank, tower take |
| 1.0-1.2 | Excited | Team fight win, dragon secure |
| 1.3-1.5 | HYPE | Baron steal, Pentakill |
| 1.6-2.0 | Unhinged | Game-winning backdoor, Worlds finals |

### CFG Weight (0.0 - 1.0)

Controls pacing and pauses. Lower = faster delivery:

```
0.0 ──────────────────────────────────────────────────── 1.0
│              │              │              │
Rapid-fire     Action         Standard       Dramatic
No pauses      Sequences      Commentary     Pauses
```

| Value | Pacing | Best For |
|-------|--------|----------|
| 0.1-0.2 | Machine gun | Chaotic teamfights, multiple kills |
| 0.3-0.4 | Quick | Skirmishes, objective contests |
| 0.5-0.6 | Natural | General play-by-play |
| 0.7-0.8 | Deliberate | Building tension, analysis |
| 0.9-1.0 | Slow, dramatic | Major moment buildup |

### Temperature (0.1 - 1.5)

Controls voice variation per synthesis:

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.3-0.5 | Consistent | Repeated callouts, UI feedback |
| 0.6-0.8 | Natural variation | Most commentary |
| 0.9-1.2 | Dynamic range | Exciting moments |
| 1.3+ | Unpredictable | Creative/experimental |

---

## Commentary Presets

Copy-paste ready presets for common scenarios:

### Pentakill / Ace

```javascript
const PENTAKILL = {
  model: 'standard',
  voice: { exaggeration: 1.4, cfg_weight: 0.2, temperature: 0.9 }
};
// "PENTAKILL! PENTAKILL! OH MY GOD!"
```

### Baron/Dragon Steal

```javascript
const OBJECTIVE_STEAL = {
  model: 'standard',
  voice: { exaggeration: 1.3, cfg_weight: 0.25, temperature: 0.85 }
};
// "STOLEN! The smite was PERFECT!"
```

### Teamfight Chaos

```javascript
const TEAMFIGHT = {
  model: 'standard',
  voice: { exaggeration: 1.1, cfg_weight: 0.2, temperature: 0.8 }
};
// "They're going in! It's absolute chaos!"
```

### Outplay Reaction

```javascript
const OUTPLAY = {
  model: 'standard',
  voice: { exaggeration: 1.2, cfg_weight: 0.3, temperature: 0.8 }
};
// "LOOK AT THE MOVES! What was that?!"
```

### Calm Analysis

```javascript
const ANALYSIS = {
  model: 'standard',
  voice: { exaggeration: 0.4, cfg_weight: 0.6, temperature: 0.6 }
};
// "Looking at the gold differential..."
```

### Quick Update (Turbo)

```javascript
const QUICK_UPDATE = {
  model: 'turbo',
  voice: { temperature: 0.7 }
};
// "First blood goes to blue side"
```

### Hype Intro

```javascript
const HYPE_INTRO = {
  model: 'standard',
  voice: { exaggeration: 1.0, cfg_weight: 0.35, temperature: 0.75 }
};
// "Welcome to the rift everyone!"
```

### Game-Ending

```javascript
const GAME_ENDING = {
  model: 'standard',
  voice: { exaggeration: 1.5, cfg_weight: 0.15, temperature: 1.0 }
};
// "AND THAT'S THE GAME! HISTORY HAS BEEN MADE!"
```

---

## Audio Playback

### Web Audio API (Browser)

```javascript
class TTSPlayer {
  constructor() {
    this.audioContext = new AudioContext({ sampleRate: 24000 });
    this.queue = [];
    this.isPlaying = false;
  }

  // Call this with each binary chunk from WebSocket
  addChunk(arrayBuffer) {
    const pcm = new Int16Array(arrayBuffer);
    const float = new Float32Array(pcm.length);

    for (let i = 0; i < pcm.length; i++) {
      float[i] = pcm[i] / 32768.0;
    }

    this.queue.push(float);

    if (!this.isPlaying) {
      this.playNext();
    }
  }

  async playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;

    // Batch chunks for smoother playback
    let totalLength = 0;
    const chunks = [];

    while (this.queue.length > 0 && totalLength < 24000) {
      const chunk = this.queue.shift();
      chunks.push(chunk);
      totalLength += chunk.length;
    }

    const combined = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    const buffer = this.audioContext.createBuffer(1, combined.length, 24000);
    buffer.getChannelData(0).set(combined);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.onended = () => this.playNext();
    source.start();
  }

  stop() {
    this.queue = [];
    this.isPlaying = false;
  }
}
```

### Node.js with Speaker

```javascript
const Speaker = require('speaker');
const WebSocket = require('ws');

const speaker = new Speaker({
  channels: 1,
  bitDepth: 16,
  sampleRate: 24000
});

const ws = new WebSocket('ws://localhost:8081/ws/tts');
ws.binaryType = 'nodebuffer';

ws.on('message', (data) => {
  if (Buffer.isBuffer(data)) {
    speaker.write(data);
  }
});
```

### Python with PyAudio

```python
import pyaudio
import websocket
import json
import threading

class TTSClient:
    def __init__(self, ws_url='ws://localhost:8081/ws/tts'):
        self.ws_url = ws_url
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )

    def on_message(self, ws, message):
        if isinstance(message, bytes):
            self.stream.write(message)
        else:
            msg = json.loads(message)
            if msg['type'] == 'done':
                print(f"Done: {msg['duration_ms']}ms")

    def speak(self, text, preset='hype'):
        presets = {
            'hype': {'exaggeration': 1.2, 'cfg_weight': 0.3, 'temperature': 0.8},
            'calm': {'exaggeration': 0.4, 'cfg_weight': 0.6, 'temperature': 0.6},
        }

        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message
        )

        def on_open(ws):
            ws.send(json.dumps({
                'type': 'say',
                'id': f'req-{time.time()}',
                'text': text,
                'model': 'standard',
                'voice': presets.get(preset, presets['hype'])
            }))

        ws.on_open = on_open
        ws.run_forever()
```

---

## Real-Time Commentary Patterns

### Queue Management

For a LoL commenter, you'll have frequent events. Use a priority queue:

```javascript
class CommentaryQueue {
  constructor(ttsClient) {
    this.tts = ttsClient;
    this.queue = [];
    this.speaking = false;
  }

  // Priority: 1 = highest (pentakill), 5 = lowest (cs update)
  add(text, priority, preset) {
    // Drop low priority if queue is full
    if (this.queue.length > 5 && priority > 3) {
      return;
    }

    this.queue.push({ text, priority, preset });
    this.queue.sort((a, b) => a.priority - b.priority);

    // Interrupt low priority speech for high priority
    if (priority === 1 && this.speaking) {
      this.tts.stop();
      this.speaking = false;
    }

    this.processQueue();
  }

  async processQueue() {
    if (this.speaking || this.queue.length === 0) return;

    this.speaking = true;
    const item = this.queue.shift();

    await this.tts.speak(item.text, item.preset);

    this.speaking = false;
    this.processQueue();
  }
}

// Usage
const queue = new CommentaryQueue(ttsClient);

queue.add('PENTAKILL!', 1, PENTAKILL);        // Immediate
queue.add('Dragon spawning soon', 4, ANALYSIS); // Can wait
queue.add('First blood!', 2, TEAMFIGHT);        // High priority
```

### Event-Driven Commentary

Map game events to commentary:

```javascript
const EVENT_HANDLERS = {
  'CHAMPION_KILL': (event) => {
    const kills = event.killer.kills;

    if (kills === 5) {
      return { text: `PENTAKILL FOR ${event.killer.name}!`, preset: PENTAKILL, priority: 1 };
    } else if (kills >= 3) {
      return { text: `${event.killer.name} is on a rampage!`, preset: OUTPLAY, priority: 2 };
    } else {
      return { text: `${event.killer.name} takes down ${event.victim.name}`, preset: QUICK_UPDATE, priority: 4 };
    }
  },

  'BUILDING_DESTROYED': (event) => {
    if (event.building === 'NEXUS') {
      return { text: 'THE NEXUS EXPLODES! GAME OVER!', preset: GAME_ENDING, priority: 1 };
    }
    return { text: `${event.team} destroys the ${event.building}`, preset: TEAMFIGHT, priority: 3 };
  },

  'OBJECTIVE_TAKEN': (event) => {
    const texts = {
      'BARON': 'Baron Nashor has been slain!',
      'DRAGON': `${event.dragon} dragon secured!`,
      'ELDER': 'ELDER DRAGON! This is huge!'
    };
    return { text: texts[event.objective], preset: OBJECTIVE_STEAL, priority: 2 };
  }
};

function handleGameEvent(event) {
  const handler = EVENT_HANDLERS[event.type];
  if (handler) {
    const { text, preset, priority } = handler(event);
    commentaryQueue.add(text, priority, preset);
  }
}
```

### Avoiding Overlap

```javascript
class SmartCommentator {
  constructor() {
    this.lastSpeakEnd = 0;
    this.minGap = 500; // ms between utterances
  }

  shouldSpeak(priority) {
    const now = Date.now();
    const timeSinceLast = now - this.lastSpeakEnd;

    // High priority always speaks
    if (priority <= 2) return true;

    // Lower priority respects gap
    return timeSinceLast > this.minGap;
  }

  onDone(msg) {
    this.lastSpeakEnd = Date.now();
  }
}
```

---

## Performance Optimization

### Model Selection Strategy

```javascript
function selectModel(eventType, queueLength) {
  // Use turbo when queue is backing up
  if (queueLength > 3) return 'turbo';

  // Use standard for important moments
  const importantEvents = ['PENTAKILL', 'BARON_STEAL', 'GAME_END'];
  if (importantEvents.includes(eventType)) return 'standard';

  // Default to turbo for speed
  return 'turbo';
}
```

### Text Optimization

```javascript
function optimizeText(text) {
  // Shorter text = faster synthesis

  // Remove filler words for turbo mode
  text = text.replace(/\b(just|really|very|actually)\b/gi, '');

  // Use contractions
  text = text.replace(/\bdo not\b/gi, "don't");
  text = text.replace(/\bcan not\b/gi, "can't");

  // Limit length for real-time
  if (text.length > 200) {
    text = text.substring(0, 200) + '...';
  }

  return text;
}
```

### Connection Management

```javascript
class ResilientConnection {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onclose = () => {
      console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    };

    this.ws.onopen = () => {
      this.reconnectDelay = 1000; // Reset on successful connect
    };
  }
}
```

---

## Error Handling

```javascript
ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
      case 'error':
        handleError(msg);
        break;
      case 'done':
        // Success
        break;
    }
  }
};

function handleError(error) {
  switch (error.code) {
    case 'TEXT_TOO_LONG':
      // Truncate and retry
      const shortened = error.text.substring(0, 4000);
      speak(shortened);
      break;

    case 'TOO_MANY_REQUESTS':
      // Back off
      setTimeout(() => speak(error.text), 1000);
      break;

    case 'SYNTH_FAILED':
      // Log and skip
      console.error('Synthesis failed:', error.message);
      break;

    default:
      console.error('Unknown error:', error);
  }
}
```

---

## Complete Example App

```javascript
// lol-commentator.js
class LoLCommentator {
  constructor(wsUrl = 'ws://localhost:8081/ws/tts') {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.player = new TTSPlayer();
    this.queue = [];
    this.speaking = false;
    this.ready = false;

    this.presets = {
      pentakill: { model: 'standard', voice: { exaggeration: 1.4, cfg_weight: 0.2, temperature: 0.9 } },
      teamfight: { model: 'standard', voice: { exaggeration: 1.1, cfg_weight: 0.25, temperature: 0.8 } },
      objective: { model: 'standard', voice: { exaggeration: 1.2, cfg_weight: 0.3, temperature: 0.8 } },
      analysis:  { model: 'standard', voice: { exaggeration: 0.4, cfg_weight: 0.6, temperature: 0.6 } },
      quick:     { model: 'turbo',    voice: { temperature: 0.7 } }
    };
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => console.log('Connected');

      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          this.player.addChunk(event.data);
        } else {
          const msg = JSON.parse(event.data);
          this.handleMessage(msg);
          if (msg.type === 'ready') {
            this.ready = true;
            resolve();
          }
        }
      };

      this.ws.onerror = reject;

      this.ws.onclose = () => {
        this.ready = false;
        setTimeout(() => this.connect(), 2000);
      };
    });
  }

  handleMessage(msg) {
    switch (msg.type) {
      case 'done':
        this.speaking = false;
        this.processQueue();
        break;
      case 'error':
        console.error('TTS Error:', msg.message);
        this.speaking = false;
        this.processQueue();
        break;
    }
  }

  say(text, presetName = 'quick', priority = 3) {
    if (!this.ready) return;

    // High priority interrupts
    if (priority === 1 && this.speaking) {
      this.player.stop();
      this.speaking = false;
    }

    this.queue.push({ text, presetName, priority });
    this.queue.sort((a, b) => a.priority - b.priority);

    // Trim queue
    if (this.queue.length > 10) {
      this.queue = this.queue.slice(0, 10);
    }

    this.processQueue();
  }

  processQueue() {
    if (this.speaking || this.queue.length === 0) return;

    const { text, presetName } = this.queue.shift();
    const preset = this.presets[presetName] || this.presets.quick;

    this.speaking = true;

    this.ws.send(JSON.stringify({
      type: 'say',
      id: `${Date.now()}`,
      text,
      model: preset.model,
      voice: preset.voice,
      audio: { format: 'pcm_s16le', sample_rate: 24000, stream: true }
    }));
  }

  // Convenience methods
  pentakill(champion) {
    this.say(`PENTAKILL! ${champion} IS UNSTOPPABLE!`, 'pentakill', 1);
  }

  kill(killer, victim) {
    this.say(`${killer} takes down ${victim}`, 'quick', 4);
  }

  objective(type, team) {
    this.say(`${type} secured by ${team}!`, 'objective', 2);
  }

  teamfight(description) {
    this.say(description, 'teamfight', 2);
  }

  analyze(text) {
    this.say(text, 'analysis', 5);
  }
}

// Usage
const commentator = new LoLCommentator();
await commentator.connect();

// Game events
commentator.pentakill('Faker');
commentator.objective('Baron Nashor', 'T1');
commentator.teamfight('Its absolute chaos in the river!');
commentator.analyze('Looking at the gold lead, T1 is in control');
```

---

## Deployment

### Docker Compose (Coolify)

```yaml
version: "3.9"
services:
  tts-service:
    build: .
    environment:
      DEVICE: cuda          # or cpu
      LOAD_STANDARD: "true"
      API_TOKEN: ${API_TOKEN}
    ports:
      - "8081:8081"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `cuda`, `cpu`, `mps`, or `auto` |
| `LOAD_STANDARD` | `true` | Load expressive model |
| `LOAD_MULTILINGUAL` | `false` | Load multi-language model |
| `API_TOKEN` | (none) | Optional auth token |
| `MAX_CONCURRENT_REQUESTS` | `3` | Per-connection limit |

---

## Tips for Great Commentary

1. **Vary your presets** - Don't use pentakill energy for everything
2. **Keep it short** - 1-2 sentences max for real-time feel
3. **Use turbo for filler** - Save standard model for highlights
4. **Queue management** - Drop low-priority items when behind
5. **Add natural phrases** - "Oh!", "Wow!", "What?!" feel authentic
6. **Match the moment** - Calm analysis after hype creates contrast

---

## Support

- GitHub Issues: https://github.com/grait-io/chatterbox-service/issues
- WebSocket Test Page: `http://localhost:8081/test.html`
- Health Check: `http://localhost:8081/health`
