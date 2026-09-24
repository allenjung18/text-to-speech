# audiobook-maker

Turns chapter text files into M4B audiobooks with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
running locally on Apple Silicon via mlx-audio. No accounts, no API keys, no per-character cost.

## Input format

A UTF-8 text file where chapters are separated by a line of 40 dashes, the first line of
each chapter is its title (containing `Chapter N` and, after `N:`, an optional subtitle), and
paragraphs are separated by blank lines.

## Setup

```bash
brew install ffmpeg espeak-ng
uv sync
```

The first run downloads the Kokoro weights (~330 MB) and a small spaCy model into the
Hugging Face cache.

## Use

```bash
# one text file -> M4B files of 25 chapters each, in ./audiobooks
uv run audiobook_maker.py make path/to/book_chapters_1_100.txt --out-dir audiobooks

# every top-level PDF in a folder whose text lives in <folder>/archived/<same stem>.txt
uv run audiobook_maker.py queue ~/Desktop/ShadowSlave

# check the parse without generating audio
uv run audiobook_maker.py make book.txt --dry-run
```

Flags: `--voice af_heart` (any Kokoro voice id), `--speed 1.0`, `--group 25`, `--limit N`,
`--title "Book Title"`, `--log-file progress.log`. `make` also takes `--chapters 1210-1261` (a
range of the file), `--split 10,10,10,10,12` (explicit chapters per M4B) and `--merge-tail`
(fold a short last group into the one before it).

## Background jobs

`audiobook.sh` runs one job detached and reports it in one line, with a macOS notification at
the end:

```bash
./audiobook.sh start 1210-1261 --group 10 --merge-tail   # finds the archived file for the range
./audiobook.sh status                                    # RUNNING 31/52 chapters, 3/5 files, ETA ~14m
./audiobook.sh wait                                      # blocks; DONE / FAILED, exit code to match
```

Every chapter is synthesized once into `<out-dir>/.cache/<hash>.wav`, keyed by model, voice,
speed, pipeline version and text. Re-runs only synthesize what changed and re-assemble the M4B
files, so new chapters or a fixed paragraph never regenerate the whole book.

Output is one `.m4b` per group with chapter markers, playable in Apple Books, BookPlayer,
Prologue, and most audiobook apps.
