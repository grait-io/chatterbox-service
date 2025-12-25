#!/bin/bash
# Build script for Chatterbox TTS Rust service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Chatterbox TTS Rust Service${NC}"
echo "========================================"

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust is not installed${NC}"
    echo "Install Rust from: https://rustup.rs/"
    exit 1
fi

# Check Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo -e "Rust version: ${YELLOW}${RUST_VERSION}${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "Python version: ${YELLOW}${PYTHON_VERSION}${NC}"

# Build mode
BUILD_MODE="${1:-release}"

if [ "$BUILD_MODE" = "release" ]; then
    echo -e "\n${GREEN}Building in release mode...${NC}"
    cargo build --release
    BINARY="target/release/chatterbox-server"
elif [ "$BUILD_MODE" = "debug" ]; then
    echo -e "\n${GREEN}Building in debug mode...${NC}"
    cargo build
    BINARY="target/debug/chatterbox-server"
else
    echo -e "${RED}Unknown build mode: ${BUILD_MODE}${NC}"
    echo "Usage: $0 [release|debug]"
    exit 1
fi

# Check if build succeeded
if [ -f "$BINARY" ]; then
    echo -e "\n${GREEN}Build successful!${NC}"
    echo -e "Binary: ${YELLOW}${BINARY}${NC}"

    # Show binary size
    SIZE=$(du -h "$BINARY" | cut -f1)
    echo -e "Size: ${YELLOW}${SIZE}${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo "To run the server:"
echo "  ./$BINARY"
echo ""
echo "Or with Docker:"
echo "  docker-compose up -d"
