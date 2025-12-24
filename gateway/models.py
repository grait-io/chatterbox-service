"""Pydantic models for WebSocket TTS messages."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Voice configuration for TTS."""
    audio_prompt_path: Optional[str] = None
    # Expressiveness parameters (work best with standard model)
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0, description="Emotional expressiveness (0.0-2.0, default 0.5)")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG weight for pacing (0.0-1.0, lower=faster)")
    temperature: float = Field(default=0.8, ge=0.1, le=1.5, description="Sampling temperature (0.1-1.5)")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0, description="Repetition penalty")


class AudioConfig(BaseModel):
    """Audio output configuration."""
    format: Literal["opus", "pcm_s16le"] = "opus"
    sample_rate: int = 24000
    stream: bool = True


class SayRequest(BaseModel):
    """Client request to synthesize speech."""
    type: Literal["say"] = "say"
    id: str = Field(..., description="Client-provided correlation ID")
    text: str = Field(..., min_length=1, max_length=5000)
    language_id: str = "en"
    model: Literal["turbo", "standard"] = Field(default="standard", description="Model: 'standard' for expressiveness, 'turbo' for speed")
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


class ReadyMessage(BaseModel):
    """Server ready message sent on connection."""
    type: Literal["ready"] = "ready"
    version: str = "1.0.0"
    models: list[str] = ["chatterbox-turbo", "chatterbox-multilingual"]
    default_format: str = "opus"


class ChunkMessage(BaseModel):
    """Optional progress message."""
    type: Literal["chunk"] = "chunk"
    id: str
    state: str = "speaking"
    offset_ms: int = 0


class DoneMessage(BaseModel):
    """Sent when utterance synthesis is complete."""
    type: Literal["done"] = "done"
    id: str
    duration_ms: int


class ErrorMessage(BaseModel):
    """Error message."""
    type: Literal["error"] = "error"
    id: Optional[str] = None
    code: str
    message: str
