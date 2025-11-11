# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Omnilingual ASR is an open-source multilingual speech recognition system supporting 1600+ languages. Built on fairseq2, it provides three model families: W2V (SSL encoders), CTC (parallel ASR), and LLM (autoregressive ASR with language conditioning and zero-shot capabilities).

## Development Commands

### Environment Setup
```bash
# Install base package
pip install -e .

# Install with data processing dependencies
pip install -e ".[data]"

# Install with development tools
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_lang_ids.py

# Run tests with verbose output
pytest -v
```

### Code Quality
```bash
# Format code (automatically fixes issues)
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/ tests/

# Linting
flake8 src/ tests/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Model Inference
```bash
# Basic transcription
python -c "
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipeline = ASRInferencePipeline(model_card='omniASR_CTC_1B')
result = pipeline.transcribe(['/path/to/audio.wav'])
print(result)
"
```

### Training
```bash
# Run training recipe
export OUTPUT_DIR="/path/to/output"
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
  --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml

# Run evaluation recipe
python -m workflows.recipes.wav2vec2.asr.eval $OUTPUT_DIR \
  --config-file workflows/recipes/wav2vec2/asr/eval/configs/fleurs-mls-mini.yaml
```

### Data Preparation
```bash
# Quick test with 2 languages (~5-10 min)
python workflows/dataprep/hf_dataset_ingestion_example.py run_short /path/to/output

# Full example with MLS + FLEURS subset (~90 min)
python workflows/dataprep/hf_dataset_ingestion_example.py run_full /path/to/output

# Verify dataloader
python -m workflows.dataprep.dataloader_example \
  --dataset_path="dataset/version=0" \
  --split="train" \
  --num_iterations=10
```

## Architecture Overview

### Model Hierarchy

Three model families built on Wav2Vec2 encoder foundation:

1. **W2V (SSL)**: Audio → Feature Extractor → Encoder → Audio Embeddings
   - Output: Contextualized embeddings (1024/1280/2048-dim)
   - Use case: Starting point for custom architectures

2. **CTC (Parallel ASR)**: W2V → Linear Projection → Vocab Logits
   - Output: Parallel predictions (9812 vocab tokens)
   - Use case: Fast on-device transcription, no language conditioning

3. **LLM (Autoregressive ASR)**: W2V → Linear Projection → Llama Decoder → Vocab Logits
   - Output: Autoregressive text generation (9812 vocab tokens)
   - Input: Audio + optional language ID (LLM variant) or 10 context examples (ZS variant)
   - Use case: Best quality, supports language conditioning and zero-shot learning

### Key Architecture Files

- **Model Implementations**:
  - `src/omnilingual_asr/models/wav2vec2_ssl/` - SSL encoder models
  - `src/omnilingual_asr/models/wav2vec2_asr/` - CTC models
  - `src/omnilingual_asr/models/wav2vec2_llama/` - LLM encoder-decoder models
    - `model.py` - Core model implementation with input validation
    - `lang_ids.py` - Language codes for 1600+ supported languages
    - `beamsearch.py` - Beam search for autoregressive decoding
    - `hub.py`, `factory.py`, `interop.py` - Model loading and interoperability

- **Inference Pipeline**: `src/omnilingual_asr/models/inference/pipeline.py`
  - Handles multiple input formats (file paths, binary data, decoded audio)
  - Supports batch processing, language conditioning, and zero-shot context

### Dataset System

The codebase uses a flexible dataset abstraction with two axes:

**Storage Backends** (where data comes from):
- `MixtureParquetStorage` - Parquet files partitioned by corpus/split/language
- `ManifestStorage` - Manifest-based storage (reference implementation)

**Task Backends** (how data is processed):
- `AsrTask` - ASR with audio+text, returns `Seq2SeqBatch`
- `SslTask` - SSL with audio-only, returns `SequenceBatch`

**Key Files**:
- `src/omnilingual_asr/datasets/impl/mixture_parquet_asr_dataset.py` - Main dataset class
- `src/omnilingual_asr/datasets/storage/` - Storage backend implementations
- `src/omnilingual_asr/datasets/tasks/` - Task backend implementations
- `src/omnilingual_asr/datasets/utils/` - Audio/text processing, batching utilities

### Asset Management System

Models, tokenizers, and datasets are managed through fairseq2's asset system using YAML cards in `src/omnilingual_asr/cards/`:

```yaml
name: omniASR_CTC_300M
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt
tokenizer_ref: omniASR_tokenizer
```

Load assets via: `load_model("omniASR_CTC_300M")` or reference by name in recipe configs.

### Training Recipes

Pre-configured workflows in `workflows/recipes/wav2vec2/asr/`:
- `recipe.py` - Main training logic
- `eval/recipe.py` - Evaluation logic
- `configs/*.yaml` - Training configurations (CTC from encoder, CTC finetune, LLM from encoder, LLM finetune)
- `dataset_selector.py` - Switches between dataset backends
- `criterion.py` - Loss functions
- `wer_calculator.py` - WER metrics computation

## Important Implementation Details

### Language Codes
Languages use format `{language_code}_{script}` (e.g., `eng_Latn`, `cmn_Hans`). The full list is in `src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py`.

### Audio Constraints
- Training data: mostly ≤30 seconds
- Inference limit: 40 seconds (hard limit in current implementation)
- Target format: 16kHz mono, normalized waveform

### Model Input Validation
`Wav2Vec2LlamaModel` validates inputs at every forward pass via `ensure_valid_forward_inputs`:
- **LLM+LID**: Requires optional `lang` field in `batch.example` (list of language codes)
- **LLM+ZS**: Requires exactly 10 context examples in `context_audio` and `context_text` fields in `batch.example`

The model behavior changes based on `batch.example` fields - always check what extra fields are present.

### Parquet Dataset Schema
```python
text: string                    # Normalized transcription
audio_bytes: list<int8>         # Compressed audio (flac/ogg), 16kHz mono
audio_size: int64              # Decoded waveform size
corpus: dictionary             # Source corpus name
split: dictionary              # train/dev/test
language: dictionary           # Language code (e.g., "deu_Latn")
```

Written with `row_group_size=100` for efficient streaming and memory management.

### Data Processing Pipeline (workflows/dataprep/)
1. Load HuggingFace datasets via Ray (or locally)
2. Apply text normalization (language-specific, via `text_tools.py`)
3. Convert audio to byte lists, resample to 16kHz (via `audio_tools.py`)
4. Write to partitioned parquet files
5. Compute corpus-language statistics for weighted sampling

### Model Downloads
Models auto-download to `~/.cache/fairseq2/assets/` on first use.

## Development Patterns

### Adding a New Model
1. Create asset card in `src/omnilingual_asr/cards/models/`
2. Define architecture config if needed (see existing model family configs)
3. Register model family/arch in model family's `config.py`
4. Test loading: `load_model("your_model_name")`

### Adding a New Dataset
1. Process data to parquet format (see `workflows/dataprep/`)
2. Create asset card in `src/omnilingual_asr/cards/datasets/`
3. Reference dataset name in training recipe config

### Extending the Inference Pipeline
The pipeline accepts three input formats (see `src/omnilingual_asr/models/inference/README.md`):
1. File paths: `["/path/to/audio.wav"]`
2. Binary data: `[open("audio.wav", "rb").read()]`
3. Decoded audio: `[{"waveform": tensor, "sample_rate": 16000}]`

All formats go through: decode → resample to 16kHz → convert to mono → normalize

## Code Style and Quality

- Python 3.10+ required
- Type hints required (strict mypy configuration)
- Code formatting: black with default settings
- Import sorting: isort with black profile
- Pre-commit hooks enforce: trailing whitespace, merge conflicts, file size limits, AST validity

## Common Issues

### Memory Issues During Training
Increase gradient accumulation in recipe config:
```yaml
trainer:
  grad_accumulation:
    num_batches: 4  # or higher
```

### Audio Length Errors
Check audio duration: must be ≤40 seconds for inference, ≤60 seconds recommended for training.

### Wrong Model Behavior with LLM Models
Verify `batch.example` fields match model expectations:
- LLM+LID needs `lang` field (optional)
- LLM+ZS needs exactly 10 `context_audio` and `context_text` examples

### Dataset Loading Fails
1. Verify parquet partitions match expected structure: `corpus=X/split=Y/language=Z/`
2. Check partition filters in dataset config
3. Verify language codes match `lang_ids.py`
