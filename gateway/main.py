"""
Chatterbox TTS WebSocket Gateway

A WebSocket server that provides streaming text-to-speech using Chatterbox models.
"""
import asyncio
import json
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.responses import JSONResponse

from models import SayRequest, ReadyMessage, DoneMessage, ErrorMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts-gateway")

# Configuration from environment
API_TOKEN = os.getenv("API_TOKEN", "")
DEVICE = os.getenv("DEVICE", "auto")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))

# Global model instances
tts_standard_model = None  # Standard model with expressiveness support
tts_turbo_model = None     # Turbo model for speed
multilingual_model = None
model_lock = asyncio.Lock()


def get_device() -> str:
    """Determine the best available device."""
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Configuration
LOAD_MULTILINGUAL = os.getenv("LOAD_MULTILINGUAL", "false").lower() == "true"
LOAD_STANDARD = os.getenv("LOAD_STANDARD", "true").lower() == "true"  # Load expressive model


def load_models():
    """Load TTS models on startup."""
    global tts_standard_model, tts_turbo_model, multilingual_model

    device = get_device()
    logger.info(f"Loading models on device: {device}")

    # Patch torch.load for MPS compatibility
    if device == "mps":
        map_location = torch.device(device)
        torch_load_original = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch_load_original(*args, **kwargs)
        torch.load = patched_torch_load

    # Load standard model with expressiveness support (recommended for commentator style)
    if LOAD_STANDARD:
        try:
            from chatterbox.tts import ChatterboxTTS
            logger.info("Loading ChatterboxTTS (standard - expressive)...")
            tts_standard_model = ChatterboxTTS.from_pretrained(device=device)
            logger.info("ChatterboxTTS loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load ChatterboxTTS: {e}")

    # Load turbo model for speed
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        logger.info("Loading ChatterboxTurboTTS...")
        tts_turbo_model = ChatterboxTurboTTS.from_pretrained(device=device)
        logger.info("ChatterboxTurboTTS loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ChatterboxTurboTTS: {e}")
        if tts_standard_model is None:
            raise  # Need at least one model

    if LOAD_MULTILINGUAL:
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            logger.info("Loading ChatterboxMultilingualTTS...")
            multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            logger.info("ChatterboxMultilingualTTS loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load ChatterboxMultilingualTTS: {e}")
    else:
        logger.info("Skipping ChatterboxMultilingualTTS (LOAD_MULTILINGUAL=false)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting TTS Gateway...")
    load_models()
    logger.info("TTS Gateway ready")
    yield
    logger.info("Shutting down TTS Gateway...")


app = FastAPI(
    title="Chatterbox TTS WebSocket Gateway",
    version="1.0.0",
    lifespan=lifespan
)


def validate_token(token: Optional[str]) -> bool:
    """Validate API token if configured."""
    if not API_TOKEN:
        return True
    return token == API_TOKEN


def audio_to_pcm_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to PCM S16LE bytes."""
    # Ensure on CPU and convert to numpy
    audio = audio_tensor.squeeze().cpu().numpy()

    # Normalize to int16 range
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16.tobytes()


def chunk_audio(audio_bytes: bytes, chunk_size_ms: int, sample_rate: int, bytes_per_sample: int = 2) -> list[bytes]:
    """Split audio into chunks of specified duration."""
    samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    chunks = []
    for i in range(0, len(audio_bytes), bytes_per_chunk):
        chunk = audio_bytes[i:i + bytes_per_chunk]
        if chunk:
            chunks.append(chunk)
    return chunks


async def synthesize_and_stream(
    websocket: WebSocket,
    request: SayRequest
) -> None:
    """Synthesize speech and stream audio chunks."""
    start_time = time.time()

    try:
        # Select model based on request and availability
        use_standard = request.model == "standard" and tts_standard_model is not None

        if request.language_id != "en" and multilingual_model is not None:
            model = multilingual_model
            model_name = "chatterbox-multilingual"
            is_standard = False
        elif use_standard:
            model = tts_standard_model
            model_name = "chatterbox-standard"
            is_standard = True
        elif tts_turbo_model is not None:
            model = tts_turbo_model
            model_name = "chatterbox-turbo"
            is_standard = False
        elif tts_standard_model is not None:
            model = tts_standard_model
            model_name = "chatterbox-standard"
            is_standard = True
        else:
            raise RuntimeError("No TTS model available")

        # Log with expressiveness params if using standard model
        if is_standard:
            logger.info(f"[{request.id}] Synthesizing with {model_name} (exag={request.voice.exaggeration}, cfg={request.voice.cfg_weight}, temp={request.voice.temperature}): {request.text[:50]}...")
        else:
            logger.info(f"[{request.id}] Synthesizing with {model_name}: {request.text[:50]}...")

        # Prepare generation kwargs
        gen_kwargs = {
            "text": request.text,
            "temperature": request.voice.temperature,
            "top_p": request.voice.top_p,
            "repetition_penalty": request.voice.repetition_penalty,
        }

        # Add expressiveness params for standard model
        if is_standard:
            gen_kwargs["exaggeration"] = request.voice.exaggeration
            gen_kwargs["cfg_weight"] = request.voice.cfg_weight

        # Add voice conditioning if specified
        if request.voice.audio_prompt_path:
            gen_kwargs["audio_prompt_path"] = request.voice.audio_prompt_path

        # Run synthesis in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        async with model_lock:
            audio_tensor = await loop.run_in_executor(
                None,
                lambda: model.generate(**gen_kwargs)
            )

        # Get sample rate from model
        sample_rate = getattr(model, 'sr', 24000)

        # Convert to PCM bytes
        audio_bytes = audio_to_pcm_bytes(audio_tensor, sample_rate)

        # Calculate duration
        duration_ms = int(len(audio_bytes) / (sample_rate * 2) * 1000)

        # Chunk and stream (40ms chunks for smooth playback)
        chunk_ms = 40
        chunks = chunk_audio(audio_bytes, chunk_ms, sample_rate)

        logger.info(f"[{request.id}] Streaming {len(chunks)} chunks ({duration_ms}ms total)")

        for chunk in chunks:
            await websocket.send_bytes(chunk)
            # Small delay to simulate streaming (can be removed for max throughput)
            await asyncio.sleep(0.001)

        # Send done message
        done_msg = DoneMessage(id=request.id, duration_ms=duration_ms)
        await websocket.send_text(done_msg.model_dump_json())

        elapsed = time.time() - start_time
        logger.info(f"[{request.id}] Complete in {elapsed:.2f}s (RTF: {elapsed / (duration_ms / 1000):.2f})")

    except Exception as e:
        logger.error(f"[{request.id}] Synthesis failed: {e}")
        error_msg = ErrorMessage(
            id=request.id,
            code="SYNTH_FAILED",
            message=str(e)
        )
        await websocket.send_text(error_msg.model_dump_json())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_loaded = []
    if tts_standard_model is not None:
        models_loaded.append("chatterbox-standard")
    if tts_turbo_model is not None:
        models_loaded.append("chatterbox-turbo")
    if multilingual_model is not None:
        models_loaded.append("chatterbox-multilingual")

    return JSONResponse({
        "status": "healthy" if models_loaded else "degraded",
        "models": models_loaded,
        "device": get_device()
    })


@app.websocket("/ws/tts")
async def websocket_tts(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for TTS streaming."""
    # Validate token
    if not validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")

    # Send ready message with available models
    available_models = []
    if tts_standard_model is not None:
        available_models.append("chatterbox-standard")
    if tts_turbo_model is not None:
        available_models.append("chatterbox-turbo")
    if multilingual_model is not None:
        available_models.append("chatterbox-multilingual")
    ready_msg = ReadyMessage(models=available_models)
    await websocket.send_text(ready_msg.model_dump_json())

    # Track active requests for this client
    active_requests = 0

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                error_msg = ErrorMessage(
                    code="INVALID_JSON",
                    message="Failed to parse JSON message"
                )
                await websocket.send_text(error_msg.model_dump_json())
                continue

            msg_type = msg.get("type")

            if msg_type == "say":
                # Validate request
                try:
                    request = SayRequest(**msg)
                except Exception as e:
                    error_msg = ErrorMessage(
                        id=msg.get("id"),
                        code="INVALID_REQUEST",
                        message=str(e)
                    )
                    await websocket.send_text(error_msg.model_dump_json())
                    continue

                # Check text length
                if len(request.text) > MAX_TEXT_LENGTH:
                    error_msg = ErrorMessage(
                        id=request.id,
                        code="TEXT_TOO_LONG",
                        message=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
                    )
                    await websocket.send_text(error_msg.model_dump_json())
                    continue

                # Check concurrent requests
                if active_requests >= MAX_CONCURRENT_REQUESTS:
                    error_msg = ErrorMessage(
                        id=request.id,
                        code="TOO_MANY_REQUESTS",
                        message=f"Maximum {MAX_CONCURRENT_REQUESTS} concurrent requests exceeded"
                    )
                    await websocket.send_text(error_msg.model_dump_json())
                    continue

                # Process request
                active_requests += 1
                try:
                    await synthesize_and_stream(websocket, request)
                finally:
                    active_requests -= 1

            else:
                error_msg = ErrorMessage(
                    code="UNKNOWN_MESSAGE_TYPE",
                    message=f"Unknown message type: {msg_type}"
                )
                await websocket.send_text(error_msg.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8081")),
        reload=False,
        log_level="info"
    )
