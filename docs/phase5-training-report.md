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
- **Configuration**:
  - `max_epochs`: 4000
  - `batch_size`: 8 (adjusted for optimal step counting)
  - `checkpoint_dir`: `train/real/checkpoints`
  - Resuming from `lessac-medium` via auto-resume script

### Midpoint Evaluation (Epoch 2361)
- **Validation Score (MOS)**: 3.4975
- **Quality Notes**:
  - Intelligible, but exhibited noticeable robotic and "staticy" audio artifacts.
  - Pacing felt slightly rushed and unnatural.
- **Action**: This is highly typical of midpoint VITS training convergence. The model requires further epochs to smooth out the metallic artifacts and properly learn the human pacing of the dataset. Resumed training targeting 4000+ epochs.
