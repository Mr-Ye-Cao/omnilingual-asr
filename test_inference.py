#!/usr/bin/env python3
"""Quick test to verify model inference works"""

import numpy as np
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

print("Loading model...")
pipeline = ASRInferencePipeline(model_card='omniASR_CTC_1B', device='cuda')
print("Model loaded successfully!")

# Create a dummy audio sample (1 second of random audio at 16kHz)
# In real usage, this would be actual audio data
sample_rate = 16000
duration = 1  # seconds
dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)

# Normalize to [-1, 1] range
dummy_audio = dummy_audio / np.abs(dummy_audio).max()

audio_input = {
    "waveform": torch.from_numpy(dummy_audio),
    "sample_rate": sample_rate
}

print("\nTesting transcription with dummy audio...")
result = pipeline.transcribe([audio_input], batch_size=1)
print(f"Transcription result: {result}")
print("\nModel is working! Ready for live transcription setup.")
