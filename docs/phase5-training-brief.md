# Phase 5 brief — train a Piper voice from the owner's recordings

You are picking this up cold. Read this whole file before running anything.

## Context

`audiobook-maker` turns text chapters into M4B audiobooks with Kokoro-82M
(mlx-audio) on Apple Silicon. See `README.md` and `CLAUDE.md`. Kokoro ships
weights only, so it cannot learn a new voice. Phase 5 fine-tunes **Piper**
(VITS, `OHF-Voice/piper1-gpl`) on the owner's own voice, starting from an
existing English checkpoint (`rhasspy/piper-checkpoints`, en_US lessac medium).
No rented GPU: training runs locally in resumable overnight chunks.

## Hard rules

- **Never commit or upload voice recordings.** They live in `voices/`, which is
  gitignored. No cloud storage, no Hugging Face upload, no pastebins.
- **Do not change the Kokoro pipeline** (`audiobook_maker.py` make/queue paths,
  `textprep.py`), and do not touch `audiobooks/` or `~/Desktop/ShadowSlave/`.
- Put Piper in its **own venv** (`train/.venv`); mlx-audio's pins in the main
  `.venv` must not move.
- Do not delete anything you did not create. Ask before downloads over 5 GB.
- Follow piper1-gpl's own `TRAINING.md` for exact flags; do not guess them.
- Stop and report at every **STOP** below.

## Step 0 — identify the machine

Print the chip, RAM and whether `torch.backends.mps.is_available()`.
- Apple Silicon: try `accelerator=mps`; fall back to CPU if an op is unsupported
  (record which op).
- Intel Mac: CPU only; the AMD GPU is not usable for training. Expect this to be
  very slow; say so in the report.

## Step 1 — smoke test on public-domain data (needs no recordings)

1. Set up `train/.venv` with piper1-gpl's training extras and espeak-ng.
2. Download the lessac-medium checkpoint and the first ~200 clips of LJSpeech
   (public domain) into `train/smoke/` (gitignored).
3. Preprocess, then fine-tune for about 20 minutes wall time.
4. Resume from the saved checkpoint for 5 more minutes, to prove chunking works.
5. Export to ONNX and synthesize one sentence to `train/smoke/test.wav`.

**STOP — report:** device used, seconds per step or epoch, peak memory, whether
resume worked, and an estimate of nights needed for 1 h of data at ~1000
fine-tune epochs. Do not continue until the owner says so.

## Step 2 — recording tool (only if the owner asks for it)

Add `./audiobook.sh record` (plus a function in a new `record.py`, not in
`audiobook_maker.py`):
- Prompts come from the CMU ARCTIC prompt list, plus the names in
  `~/Desktop/ShadowSlave/lexicon.json` spoken in short sentences.
- Show one line; record from the Mac's default mic at 22050 Hz mono; Enter
  keeps it, `r` redoes it, `q` quits. Resume at the next unrecorded line.
- Write `voices/dataset/wavs/<id>.wav` and append `<id>|<text>` to
  `voices/dataset/metadata.csv` (the LJSpeech layout Piper reads).
- After a session, run a Whisper check (reuse `check`'s mlx-whisper setup):
  flag clips with WER > 10%, clipping (peak > -1 dBFS) or high noise, and list
  them for re-recording.

Add tests under `tests/` for the metadata writing and resume logic.

## Step 3 — real fine-tune (only when `voices/dataset` holds ≥ 30 min)

Same recipe as step 1 on `voices/dataset`, checkpointing every ~30 minutes and
run detached (`nohup` or `caffeinate -i`), one night per chunk. After each
night: export ONNX, render `audiobooks/compare/phase4/passage.txt`, and
round-trip it through Whisper (WER) for the owner to judge by ear.

## Done means

A report with measured numbers (not estimates dressed as results), the
commands to resume, and the list of files created. Commit code only (no audio,
no checkpoints) with a message starting `audiobook-maker:`.
