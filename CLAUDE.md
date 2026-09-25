# audiobook-maker — for agents

Generate audiobooks with `./audiobook.sh`; do not read the source to run a job.

```bash
./audiobook.sh start 1210-1261 --split 10,10,10,10,12   # or --group 10 --merge-tail; returns after the plan line
./audiobook.sh wait      # run in the background; one line at the end: DONE: ... | FAILED (exit n) ...
./audiobook.sh status    # one line, only if the user asks mid-run
./audiobook.sh tail 20   # only after FAILED
```

- Input: `~/Desktop/ShadowSlave/archived/*_chapters_<A>_<B>.txt`; a range must sit inside one file.
- Output: `~/Desktop/ShadowSlave/audiobooks/Shadow Slave - Ch <first>-<last>.m4b`. A macOS notification fires at the end.
- One job at a time (the script refuses a second). Chapters are cached, so reruns only rebuild the M4Bs.
- Cost: ~40 s per chapter on the M4 Pro (~10x realtime); `start` prints an estimate from the cache's own timings.
- Check a split before a long run: add `--dry-run` to `start` (seconds, no audio).
- Narration checks: `./audiobook.sh oov N-M` (lexicon candidates, seconds), `./audiobook.sh check N-M` (Whisper, ~15 s/chapter, refuses while a render runs; reports in `audiobooks/check/`). Lexicon: `~/Desktop/ShadowSlave/lexicon.json`. Tests: `uv run pytest`.
- Default voice is am_liam (the user's pick); chapters 1162-1261 were rendered in af_heart on purpose.
- Other flags pass through to `audiobook_maker.py make`: `--voice af_heart`, `--speed 1.0`, `--title`.

## Key Learnings & Mac M4 Pro Training Quirks
- **App Nap & Throttling**: macOS aggressively suspends background python processes (PyTorch Lightning) when the display sleeps. This drops training speed from 1-3 mins/epoch to 15+ mins/epoch or deadlocks entirely. **Fix**: Wrap training jobs in `caffeinate -i` (e.g., `caffeinate -i python run_piper_final.py`).
- **Piper Fine-Tuning Convergence**: When fine-tuning a 30-minute high-quality voice dataset over a base model (like `lessac-medium`), Epochs 2000-2500 often produce a highly intelligible but "robotic/staticy" voice with rushed pacing. This is the midpoint. True human-like inflection and noise-reduction convergence typically requires 4000-5000+ epochs.
- **Validation Metric**: Piper uses an automatic MOS (Mean Opinion Score) predictor (`val_mos`). A score in the mid-3s (e.g., 3.2 - 3.5) is extremely high for this strict predictor.
- **Auto-Resume**: PyTorch Lightning training is natively resumable. Using a script to regex the highest epoch out of `train/real/checkpoints/lightning_logs/*/checkpoints/*.ckpt` allows for seamless start/stop.
- **Aliases**: We created project-specific terminal shortcuts (`train` and `progress`) inside `~/.zshrc` that map to `audiobook.sh train` and `audiobook.sh progress`.
