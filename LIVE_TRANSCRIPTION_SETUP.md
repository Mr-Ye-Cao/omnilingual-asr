# Live Audio Transcription Setup

## Overview
This setup allows you to stream audio from your Mac to your home server (RTX 5090) for real-time transcription.

**Server**: 192.168.1.87:5000
**Model**: omniASR_CTC_1B (on GPU)

## Server Setup (Already Running!)

The server is currently running on your home server at:
- Local: http://127.0.0.1:5000
- Network: http://192.168.1.87:5000

### Server Management

**Check if server is running:**
```bash
curl http://192.168.1.87:5000/health
```

**Start the server:**
```bash
cd /home/ye/ml-experiments/omnilingual-asr
conda activate omni-asr
python asr_server.py
```

**Stop the server:**
```bash
# Press Ctrl+C in the terminal where it's running
# Or find and kill the process:
ps aux | grep asr_server
kill <PID>
```

## Mac Client Setup

### 1. Install Dependencies on Your Mac

```bash
# Install Python dependencies
pip install sounddevice numpy requests

# If you don't have pip, install it first:
# python3 -m ensurepip --upgrade
```

### 2. Copy the Client Script to Your Mac

Copy `mac_client.py` from the server to your Mac:

```bash
# On your Mac:
scp ye@192.168.1.87:/home/ye/ml-experiments/omnilingual-asr/mac_client.py ~/
```

Or manually create the file on your Mac with the contents of `mac_client.py`.

### 3. Update the Server URL

Edit `mac_client.py` on your Mac and update the SERVER_URL:

```python
SERVER_URL = "http://192.168.1.87:5000"  # Your home server IP
```

### 4. Run the Client

```bash
# On your Mac:
python3 mac_client.py
```

## Usage

Once the client is running:
1. It will capture audio from your Mac's microphone
2. Every 3 seconds, it sends an audio chunk to the server
3. The server transcribes it and returns the text
4. The transcription is displayed in your terminal

**Press Ctrl+C to stop**

## Configuration Options

### Adjust Chunk Duration

In `mac_client.py`, change:
```python
CHUNK_DURATION = 3  # seconds per chunk (try 2-5 seconds)
```

- Shorter chunks = faster response, but less context
- Longer chunks = more context, but slower updates
- Max: 40 seconds (model limit)

### Add Language Hint

For better accuracy, specify the language. In `mac_client.py`, modify the `send_audio_for_transcription` call:

```python
transcription = send_audio_for_transcription(audio_data, lang="eng_Latn")  # English
```

Available languages in: `src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py`

### Silence Detection Threshold

Adjust the silence detection threshold in `mac_client.py`:
```python
if rms > 0.01:  # Lower = more sensitive, Higher = less sensitive
```

## API Endpoints

### Health Check
```bash
curl http://192.168.1.87:5000/health
```

### Transcribe (JSON)
```bash
curl -X POST http://192.168.1.87:5000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "<base64-encoded-audio>",
    "sample_rate": 16000,
    "format": "float32"
  }'
```

### Transcribe (Binary)
```bash
curl -X POST "http://192.168.1.87:5000/transcribe_binary?sample_rate=16000" \
  --data-binary @audio.raw
```

## Performance Notes

- **Latency**: ~200-500ms for transcription on RTX 5090
- **Audio Format**: 16kHz mono, float32 PCM
- **Max Audio Length**: 40 seconds per chunk
- **Recommended Chunk Size**: 2-5 seconds for low latency

## Troubleshooting

### "Cannot connect to server"
- Check server is running: `curl http://192.168.1.87:5000/health`
- Check firewall: `sudo ufw status` (on server)
- Verify network connectivity: `ping 192.168.1.87`

### "No module named sounddevice" on Mac
```bash
pip install sounddevice
```

### High latency
- Reduce CHUNK_DURATION (e.g., to 2 seconds)
- Check network speed: `ping -c 10 192.168.1.87`

### Poor transcription quality
- Speak clearly and closer to the microphone
- Reduce background noise
- Increase CHUNK_DURATION for more context
- Add language hint with `lang` parameter

## Alternative: Use Faster Model

To use the CTC 300M model (faster, less accurate):

In `asr_server.py`, change:
```python
pipeline = ASRInferencePipeline(model_card='omniASR_CTC_300M', device='cuda')
```

Or use LLM model with language conditioning (slower, more accurate):
```python
pipeline = ASRInferencePipeline(model_card='omniASR_LLM_1B', device='cuda')
```

## Files

- **Server**: `asr_server.py` - HTTP server for transcription
- **Client**: `mac_client.py` - Mac audio capture and streaming client
- **Test**: `test_server.py` - Test script to verify server is working
- **Test Inference**: `test_inference.py` - Test model loading and inference

## Next Steps

1. **Copy `mac_client.py` to your Mac**
2. **Install dependencies on Mac**: `pip install sounddevice numpy requests`
3. **Update SERVER_URL** in `mac_client.py` to `http://192.168.1.87:5000`
4. **Run**: `python3 mac_client.py`
5. **Speak** and watch the transcriptions appear!

Enjoy your low-latency live transcription system! 🎤→📝
