#!/usr/bin/env python3
"""
Live ASR Transcription Server
Receives audio chunks via HTTP and returns transcriptions
"""

import io
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global model instance - load once on startup
print("Loading ASR model...")
pipeline = ASRInferencePipeline(model_card='omniASR_CTC_1B', device='cuda')
print("Model loaded! Server ready.")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": "omniASR_CTC_1B"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio from POST request

    Accepts:
    - 'audio': base64-encoded audio data or raw bytes
    - 'sample_rate': sample rate (default: 16000)
    - 'format': 'float32' or 'int16' (default: 'float32')
    - 'lang': optional language code (e.g., 'eng_Latn')

    Returns:
    - JSON with 'transcription' field
    """
    try:
        data = request.json

        # Get audio data
        audio_b64 = data.get('audio')
        sample_rate = data.get('sample_rate', 16000)
        audio_format = data.get('format', 'float32')
        lang = data.get('lang', None)

        if not audio_b64:
            return jsonify({"error": "No audio data provided"}), 400

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)

        # Convert to numpy array based on format
        if audio_format == 'float32':
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        elif audio_format == 'int16':
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            return jsonify({"error": f"Unsupported format: {audio_format}"}), 400

        # Check audio length (max 40 seconds as per model limits)
        duration = len(audio_array) / sample_rate
        if duration > 40:
            return jsonify({"error": f"Audio too long: {duration:.1f}s (max 40s)"}), 400

        # Prepare audio input
        audio_input = {
            "waveform": torch.from_numpy(audio_array),
            "sample_rate": sample_rate
        }

        # Transcribe
        transcribe_kwargs = {"batch_size": 1}
        if lang:
            transcribe_kwargs["lang"] = [lang]

        result = pipeline.transcribe([audio_input], **transcribe_kwargs)

        return jsonify({
            "transcription": result[0],
            "duration": duration,
            "sample_rate": sample_rate
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_binary', methods=['POST'])
def transcribe_binary():
    """
    Transcribe audio from raw binary POST data
    Expects raw float32 PCM audio at 16kHz
    Optional query params: sample_rate, lang
    """
    try:
        # Get binary audio data
        audio_bytes = request.data
        sample_rate = int(request.args.get('sample_rate', 16000))
        lang = request.args.get('lang', None)

        if len(audio_bytes) == 0:
            return jsonify({"error": "No audio data provided"}), 400

        # Convert to numpy array (assuming float32)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Check audio length
        duration = len(audio_array) / sample_rate
        if duration > 40:
            return jsonify({"error": f"Audio too long: {duration:.1f}s (max 40s)"}), 400

        # Prepare audio input
        audio_input = {
            "waveform": torch.from_numpy(audio_array),
            "sample_rate": sample_rate
        }

        # Transcribe
        transcribe_kwargs = {"batch_size": 1}
        if lang:
            transcribe_kwargs["lang"] = [lang]

        result = pipeline.transcribe([audio_input], **transcribe_kwargs)

        return jsonify({
            "transcription": result[0],
            "duration": duration,
            "sample_rate": sample_rate
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on all interfaces, port 5000
    # For production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
