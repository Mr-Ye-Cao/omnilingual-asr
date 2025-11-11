#!/usr/bin/env python3
"""Test the ASR server with a synthetic audio signal"""

import base64
import numpy as np
import requests

# Generate a 2-second audio signal (sine wave at 440 Hz)
sample_rate = 16000
duration = 2
t = np.linspace(0, duration, int(sample_rate * duration))
# Create a more interesting signal - mix of frequencies
audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
audio = audio.astype(np.float32)

# Encode as base64
audio_bytes = audio.tobytes()
audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# Send to server
payload = {
    "audio": audio_b64,
    "sample_rate": sample_rate,
    "format": "float32"
}

print("Sending test audio to server...")
response = requests.post("http://localhost:5000/transcribe", json=payload)

if response.status_code == 200:
    result = response.json()
    print(f"✓ Success!")
    print(f"  Transcription: '{result.get('transcription', '')}'")
    print(f"  Duration: {result.get('duration', 0):.2f}s")
else:
    print(f"✗ Error: {response.status_code}")
    print(f"  {response.text}")
