# Phase 5: Voice Training Report

## Step 0 & 1: Smoke Test
- **Status**: Completed Successfully.
- **Hardware**: M4 Pro Apple Silicon (MPS enabled).
- **Details**: `piper-tts` installed correctly in isolated virtual environment. The 10-epoch fine-tune test using the LJSpeech slice completed without errors.

## Step 2: Recording & Dataset Curation
- **Status**: Completed Successfully.
- **Tool**: Built `record.py` and linked to `./audiobook.sh record`.
- **Customization**: Included Whisper verification and auto-slicing functionality.
- **Dataset Source**: Pure Gemini voice (via iOS Voice Memos & Screen Recordings).
- **Filtration**: Auto-translated to english via Whisper; manually scrubbed 20 human testing clips and 5 non-English (Korean/Japanese) clips to guarantee 100% voice purity.
- **Final Metrics**:
  - **Total Clips**: 647
  - **Total Duration**: 30 minutes, 19 seconds

## Step 3: Real Fine-Tune
- **Status**: Currently Running.
- **Command**: `train/.venv/bin/python run_piper_final.py`
- **Configuration**:
  - `max_epochs`: 4000
  - `batch_size`: 32
  - `checkpoint_dir`: `train/real`
  - Resuming from `lessac-medium`
- **Expected Completion**: Training will run overnight and log checkpoints to `train/real/checkpoints`.
