# ============================================================================
# Chatterbox TTS - Rust Inference Server
# Multi-stage build for optimized production image
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Rust Builder
# -----------------------------------------------------------------------------
FROM rust:1.76-bookworm AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy Cargo files first for better caching
COPY rust-tts/Cargo.toml rust-tts/Cargo.lock* ./

# Create dummy source for dependency caching
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub mod config; pub mod error; pub mod models; pub mod audio; pub mod server; pub mod python;' > src/lib.rs && \
    touch src/config.rs src/error.rs src/models.rs src/audio.rs src/server.rs src/python.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release || true

# Copy actual source
COPY rust-tts/src ./src

# Build the actual binary
RUN cargo build --release

# -----------------------------------------------------------------------------
# Stage 2: Python Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 app

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Chatterbox source
COPY src/chatterbox /app/src/chatterbox

# Set Python path
ENV PYTHONPATH=/app/src

# -----------------------------------------------------------------------------
# Stage 3: Production Image
# -----------------------------------------------------------------------------
FROM python-base AS production

# Copy Rust binary
COPY --from=rust-builder /build/target/release/chatterbox-server /usr/local/bin/

# Create directories
RUN mkdir -p /app/voices /app/.cache/huggingface && \
    chown -R app:app /app

# Switch to app user
USER app

# Environment variables
ENV HOST=0.0.0.0 \
    PORT=8081 \
    DEVICE=auto \
    LOAD_STANDARD=true \
    LOAD_TURBO=false \
    LOAD_MULTILINGUAL=false \
    MAX_CONCURRENT_REQUESTS=3 \
    MAX_TEXT_LENGTH=5000 \
    VOICES_DIR=/app/voices \
    HF_HOME=/app/.cache/huggingface \
    CHATTERBOX_SRC=/app/src \
    RUST_LOG=chatterbox_tts=info,tower_http=info

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run server
CMD ["chatterbox-server"]

# -----------------------------------------------------------------------------
# Stage 4: GPU Production Image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS gpu-production

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create app user
RUN useradd -m -u 1000 app

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Chatterbox source
COPY src/chatterbox /app/src/chatterbox

# Copy Rust binary
COPY --from=rust-builder /build/target/release/chatterbox-server /usr/local/bin/

# Create directories
RUN mkdir -p /app/voices /app/.cache/huggingface && \
    chown -R app:app /app

# Switch to app user
USER app

# Environment variables
ENV HOST=0.0.0.0 \
    PORT=8081 \
    DEVICE=cuda \
    LOAD_STANDARD=true \
    LOAD_TURBO=false \
    LOAD_MULTILINGUAL=false \
    MAX_CONCURRENT_REQUESTS=3 \
    MAX_TEXT_LENGTH=5000 \
    VOICES_DIR=/app/voices \
    HF_HOME=/app/.cache/huggingface \
    CHATTERBOX_SRC=/app/src \
    PYTHONPATH=/app/src \
    RUST_LOG=chatterbox_tts=info,tower_http=info

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run server
CMD ["chatterbox-server"]
