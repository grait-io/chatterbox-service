#!/usr/bin/env python3
"""
WebSocket TTS Test Client

Usage:
    python test_ws_client.py "Hello, this is a test message"
    python test_ws_client.py --url ws://localhost:8081/ws/tts "Test message"
    python test_ws_client.py --output test.wav "Generate and save to file"
"""
import argparse
import asyncio
import json
import struct
import sys
import wave
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


async def test_tts(url: str, text: str, output_file: str | None = None, token: str | None = None):
    """Connect to TTS WebSocket and synthesize speech."""
    # Add token to URL if provided
    if token:
        url = f"{url}?token={token}"

    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        # Wait for ready message
        ready_data = await ws.recv()
        ready_msg = json.loads(ready_data)
        print(f"Server ready: {ready_msg}")

        if ready_msg.get("type") != "ready":
            print(f"Unexpected message: {ready_msg}")
            return

        # Send say request
        request_id = f"test-{datetime.now().strftime('%H%M%S')}"
        say_msg = {
            "type": "say",
            "id": request_id,
            "text": text,
            "language_id": "en",
            "audio": {
                "format": "pcm_s16le",
                "sample_rate": 24000,
                "stream": True
            }
        }

        print(f"Sending request: {text[:50]}...")
        await ws.send(json.dumps(say_msg))

        # Collect audio chunks
        audio_chunks = []
        total_bytes = 0

        while True:
            msg = await ws.recv()

            if isinstance(msg, bytes):
                # Binary audio data
                audio_chunks.append(msg)
                total_bytes += len(msg)
                print(f"  Received audio chunk: {len(msg)} bytes (total: {total_bytes})", end="\r")
            else:
                # JSON control message
                ctrl_msg = json.loads(msg)
                print(f"\nControl message: {ctrl_msg}")

                if ctrl_msg.get("type") == "done":
                    print(f"Synthesis complete! Duration: {ctrl_msg.get('duration_ms')}ms")
                    break
                elif ctrl_msg.get("type") == "error":
                    print(f"Error: {ctrl_msg.get('message')}")
                    return

        # Combine audio chunks
        audio_data = b"".join(audio_chunks)
        print(f"\nTotal audio: {len(audio_data)} bytes ({len(audio_data) / (24000 * 2):.2f}s)")

        # Save to WAV file if requested
        if output_file:
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)
            print(f"Saved to: {output_file}")

        return audio_data


def main():
    parser = argparse.ArgumentParser(description="Test TTS WebSocket Service")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8081/ws/tts", help="WebSocket URL")
    parser.add_argument("--output", "-o", help="Output WAV file path")
    parser.add_argument("--token", help="API token for authentication")

    args = parser.parse_args()

    asyncio.run(test_tts(args.url, args.text, args.output, args.token))


if __name__ == "__main__":
    main()
