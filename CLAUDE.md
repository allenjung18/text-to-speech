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
- Other flags pass through to `audiobook_maker.py make`: `--voice af_heart`, `--speed 1.0`, `--title`.
