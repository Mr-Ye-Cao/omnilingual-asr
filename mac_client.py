#!/usr/bin/env python3
"""
Mac Audio Streaming Client for Live Transcription
Captures audio from microphone and streams to ASR server
"""

import sys
import base64
import time
import numpy as np
import requests
import sounddevice as sd
import queue
import threading

# Configuration
SERVER_URL = "http://localhost:5000"  # Change to your server IP if different
SAMPLE_RATE = 16000  # 16kHz for the model
CHUNK_DURATION = 3  # seconds per chunk
CHANNELS = 1  # mono audio
DTYPE = np.float32

# Audio queue for thread-safe communication
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback for audio input stream"""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    # Put audio data in queue
    audio_queue.put(indata.copy())

def send_audio_for_transcription(audio_data, lang=None):
    """Send audio chunk to server for transcription"""
    try:
        # Convert to bytes
        audio_bytes = audio_data.tobytes()

        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Prepare request
        payload = {
            "audio": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "format": "float32"
        }
        if lang:
            payload["lang"] = lang

        # Send to server
        response = requests.post(
            f"{SERVER_URL}/transcribe",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("transcription", "")
        else:
            print(f"Error: {response.status_code} - {response.text}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Error sending audio: {e}", file=sys.stderr)
        return None

def main():
    print("=" * 60)
    print("Live Audio Transcription Client")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Chunk Duration: {CHUNK_DURATION} seconds")
    print("=" * 60)

    # Check server health
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code == 200:
            print(f"✓ Server is ready: {health.json()}")
        else:
            print("✗ Server health check failed!")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print(f"  Make sure the server is running at {SERVER_URL}")
        return

    print("\nStarting audio capture...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    # Buffer for accumulating audio
    audio_buffer = []
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
        ):
            while True:
                # Get audio data from queue
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    audio_buffer.append(chunk)

                    # Calculate total samples
                    total_samples = sum(len(c) for c in audio_buffer)

                    # If we have enough audio, process it
                    if total_samples >= samples_per_chunk:
                        # Concatenate all chunks
                        audio_data = np.concatenate(audio_buffer, axis=0)

                        # Take only the required samples
                        audio_data = audio_data[:samples_per_chunk].flatten()

                        # Clear buffer for next chunk
                        audio_buffer = []

                        # Calculate audio level (RMS)
                        rms = np.sqrt(np.mean(audio_data ** 2))

                        # Only process if there's actual audio (not just silence)
                        if rms > 0.01:  # threshold for silence detection
                            print(f"[{time.strftime('%H:%M:%S')}] Processing {CHUNK_DURATION}s chunk (RMS: {rms:.4f})...")

                            # Send for transcription
                            transcription = send_audio_for_transcription(audio_data)

                            if transcription:
                                print(f"  → {transcription}")
                                print()
                            elif transcription == "":
                                print(f"  → (no speech detected)")
                                print()
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] Silence detected, skipping...")

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)

if __name__ == '__main__':
    # Check if sounddevice is installed
    try:
        import sounddevice
    except ImportError:
        print("Error: sounddevice is not installed.")
        print("Install it with: pip install sounddevice")
        sys.exit(1)

    main()
