#!/usr/bin/env python3
"""
Live Transcription Client - Optimized for Interview/Playback Audio
Tuned for longer utterances and natural pauses in interviews
"""

import sys
import base64
import time
import numpy as np
import requests
import sounddevice as sd
import queue
import threading
import webrtcvad
import struct

# Configuration
SERVER_URL = "http://localhost:5000"
SAMPLE_RATE = 16000  # 16kHz for the model
CHANNELS = 1  # mono audio
DTYPE = np.float32

# VAD Configuration - Optimized for Interview Audio
VAD_AGGRESSIVENESS = 1  # 0-3: Lower = more sensitive (better for speaker audio)
VAD_FRAME_DURATION = 30  # ms - webrtcvad supports 10, 20, or 30
SILENCE_DURATION = 1.5  # seconds of silence before ending utterance (longer for interviews)
MIN_UTTERANCE_DURATION = 1.0  # minimum seconds of speech to process (filter short artifacts)
MAX_UTTERANCE_DURATION = 30  # max seconds (model supports up to 40s)

# Streaming partial results
SHOW_PARTIAL_EVERY = 5  # Show partial transcription every N seconds while speaking

# Language - Set to None for auto-detect, or specify language code
# Examples: "jpn" (Japanese), "eng" (English), "cmn" (Mandarin), "spa" (Spanish)
LANGUAGE = "jpn"  # Japanese for interview

# Audio queue for thread-safe communication
audio_queue = queue.Queue()

# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def audio_callback(indata, frames, time_info, status):
    """Callback for audio input stream"""
    if status:
        print(f"⚠️  Audio status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def float32_to_int16(audio_float32):
    """Convert float32 audio to int16 for VAD"""
    audio_int16 = np.clip(audio_float32 * 32768, -32768, 32767).astype(np.int16)
    return audio_int16

def is_speech(audio_frame_int16):
    """Check if audio frame contains speech using WebRTC VAD"""
    try:
        # VAD expects bytes in int16 format
        audio_bytes = struct.pack(f'{len(audio_frame_int16)}h', *audio_frame_int16)
        return vad.is_speech(audio_bytes, SAMPLE_RATE)
    except Exception as e:
        # If VAD fails, assume it's speech to be safe
        return True

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
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("transcription", "")
        else:
            return None

    except Exception as e:
        print(f"\n⚠️  Error: {e}", file=sys.stderr)
        return None

def main():
    print("=" * 70)
    print("Live Transcription - Interview/Playback Mode")
    print("=" * 70)
    print(f"Server: {SERVER_URL}")
    print(f"Language: {LANGUAGE if LANGUAGE else 'Auto-detect'}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"VAD Aggressiveness: {VAD_AGGRESSIVENESS}/3 (optimized for speaker audio)")
    print(f"Silence Threshold: {SILENCE_DURATION}s (longer for interview pauses)")
    print(f"Min Utterance: {MIN_UTTERANCE_DURATION}s")
    print("=" * 70)

    # Check server health
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code == 200:
            print(f"✓ Server ready: {health.json()}")
        else:
            print("✗ Server health check failed!")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return

    print("\n🎤 Listening... Optimized for interview/playback audio.")
    print("   Press Ctrl+C to stop.\n")

    # State management
    is_speaking = False
    utterance_buffer = []  # Accumulates audio during speech
    silence_frames = 0
    speech_start_time = None
    last_partial_time = None

    frame_samples = int(SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    silence_frames_threshold = int((SILENCE_DURATION * 1000) / VAD_FRAME_DURATION)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
            blocksize=frame_samples
        ):
            while True:
                try:
                    # Get audio data from queue
                    chunk = audio_queue.get(timeout=0.1)
                    chunk_flat = chunk.flatten()

                    # Convert to int16 for VAD
                    chunk_int16 = float32_to_int16(chunk_flat)

                    # Check if this frame contains speech
                    has_speech = is_speech(chunk_int16)

                    if has_speech:
                        silence_frames = 0

                        if not is_speaking:
                            # Speech just started
                            is_speaking = True
                            speech_start_time = time.time()
                            last_partial_time = speech_start_time
                            utterance_buffer = [chunk_flat]
                            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] 🎙️  ")
                            sys.stdout.flush()
                        else:
                            # Continue accumulating speech
                            utterance_buffer.append(chunk_flat)

                            # Check if we should show partial results
                            current_time = time.time()
                            elapsed = current_time - speech_start_time

                            # Show partial every N seconds or if approaching max duration
                            if (elapsed - (last_partial_time - speech_start_time)) >= SHOW_PARTIAL_EVERY or elapsed >= MAX_UTTERANCE_DURATION - 1:
                                # Send partial transcription
                                audio_partial = np.concatenate(utterance_buffer)

                                if elapsed >= MAX_UTTERANCE_DURATION:
                                    # Hit max duration, finalize this utterance
                                    sys.stdout.write(f" [{elapsed:.1f}s]\n")
                                    sys.stdout.flush()

                                    transcription = send_audio_for_transcription(audio_partial, LANGUAGE)

                                    if transcription:
                                        print(f"   → {transcription}\n")
                                    else:
                                        print(f"   → (no transcription)\n")

                                    # Reset for next utterance
                                    is_speaking = False
                                    utterance_buffer = []
                                    silence_frames = 0
                                else:
                                    # Show partial progress
                                    sys.stdout.write(".")
                                    sys.stdout.flush()
                                    last_partial_time = current_time

                    else:  # Silence
                        if is_speaking:
                            silence_frames += 1

                            # Continue accumulating (might be brief pause)
                            utterance_buffer.append(chunk_flat)

                            # Check if silence lasted long enough to end utterance
                            if silence_frames >= silence_frames_threshold:
                                elapsed = time.time() - speech_start_time

                                # Only process if utterance is long enough
                                if elapsed >= MIN_UTTERANCE_DURATION:
                                    # Trim trailing silence
                                    num_silence_samples = silence_frames * frame_samples
                                    audio_data = np.concatenate(utterance_buffer)

                                    if len(audio_data) > num_silence_samples:
                                        audio_data = audio_data[:-num_silence_samples]

                                    sys.stdout.write(f" [{elapsed:.1f}s]\n")
                                    sys.stdout.flush()

                                    # Send for transcription
                                    transcription = send_audio_for_transcription(audio_data, LANGUAGE)

                                    if transcription:
                                        print(f"   → {transcription}\n")
                                    else:
                                        print(f"   → (empty)\n")
                                else:
                                    sys.stdout.write(f" [too short: {elapsed:.1f}s]\n")
                                    sys.stdout.flush()

                                # Reset state
                                is_speaking = False
                                utterance_buffer = []
                                silence_frames = 0

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
    except Exception as e:
        print(f"\n\n⚠️  Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Check dependencies
    try:
        import webrtcvad
        import sounddevice
        import numpy
        import requests
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install webrtcvad sounddevice numpy requests")
        sys.exit(1)

    main()
