# Chatterbox TTS WebSocket Service
# Single container with gateway + TTS models

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (CPU version for base, override for GPU)
ARG TORCH_VERSION=cpu
RUN if [ "$TORCH_VERSION" = "cpu" ]; then \
        pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$TORCH_VERSION" = "cuda" ]; then \
        pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install torch==2.6.0 torchaudio==2.6.0; \
    fi

# Copy and install chatterbox package
COPY pyproject.toml .
COPY src/ ./src/

# Install chatterbox dependencies (excluding torch since we already installed it)
RUN pip install numpy>=1.24.0,\<1.26.0 librosa==0.11.0 s3tokenizer transformers==4.46.3 \
    diffusers==0.29.0 resemble-perth==1.0.1 conformer==0.3.2 safetensors==0.5.3 \
    spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf

# Install chatterbox in editable mode (skip deps since we installed them)
RUN pip install -e . --no-deps

# Install gateway dependencies
COPY gateway/requirements.txt ./gateway/
RUN pip install -r gateway/requirements.txt

# Copy gateway code
COPY gateway/ ./gateway/

# Create voices directory for voice cloning
RUN mkdir -p /app/voices

# Expose ports
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8081/health').raise_for_status()"

# Default environment variables
ENV DEVICE=auto \
    PORT=8081 \
    MAX_CONCURRENT_REQUESTS=3 \
    MAX_TEXT_LENGTH=5000

# Run the gateway
WORKDIR /app/gateway
CMD ["python", "main.py"]
