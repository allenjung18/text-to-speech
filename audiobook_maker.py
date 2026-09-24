#!/usr/bin/env python3
"""audiobook_maker — turn chapter text files into M4B audiobooks with Kokoro TTS.

Runs fully local on Apple Silicon (mlx-audio). Input is a text file where
chapters are separated by a line of 40 dashes, the first line of each chapter
is its title, and paragraphs are separated by blank lines — the format the
ShadowSlave download script writes.

    audiobook make archived/shadow_slave_chapters_1162_1261.txt --out-dir audiobooks
    audiobook queue ~/Desktop/ShadowSlave        # every top-level PDF whose text is in archived/

Every chapter is synthesized once and cached by a hash of (model, voice, speed,
pipeline version, text). Re-runs only synthesize what changed, then re-assemble
the M4B files (one per --group chapters, default 25) with chapter markers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
MODEL_ID = "mlx-community/Kokoro-82M-bf16"
PIPELINE_VERSION = "1"  # bump when normalization or pause rules change: invalidates the cache
SEPARATOR = re.compile(r"^-{40,}\s*$", re.M)

PAUSE_AFTER_TITLE = 1.0
PAUSE_BETWEEN_PARAGRAPHS = 0.45
PAUSE_WITHIN_PARAGRAPH = 0.15  # between segments the pipeline splits a long paragraph into
PAUSE_END_OF_CHAPTER = 1.5


# ----------------------------------------------------------------------------- parsing

@dataclass
class Chapter:
    number: int
    subtitle: str
    paragraphs: list[str]

    @property
    def spoken_title(self) -> str:
        return f"Chapter {self.number}. {self.subtitle}" if self.subtitle else f"Chapter {self.number}"

    @property
    def marker_title(self) -> str:
        return f"Chapter {self.number}: {self.subtitle}" if self.subtitle else f"Chapter {self.number}"

    @property
    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)

    def cache_text(self) -> str:
        return "\n".join([self.spoken_title, *self.paragraphs])


def normalize(s: str) -> str:
    s = s.replace(" ", " ").replace("​", "").replace("﻿", "")
    s = s.replace("--", "—")  # ASCII double dash -> em dash (Kokoro reads it as a pause)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def parse_chapters(path: Path) -> list[Chapter]:
    raw = path.read_text(encoding="utf-8")
    chapters: list[Chapter] = []
    for block in SEPARATOR.split(raw):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        paragraphs = [normalize(p.replace("\n", " ")) for p in re.split(r"\n\s*\n", body)]
        paragraphs = [p for p in paragraphs if p]
        m = re.search(r"Chapter\s+(\d+)", title, re.I)
        if m:
            number = int(m.group(1))
        else:
            number = chapters[-1].number + 1 if chapters else 1
        # Title lines look like "<Book> Chapter N - Chapter N - N: Subtitle"; keep the subtitle.
        sm = re.search(r"\d+\s*:\s*(.+)$", title)
        subtitle = normalize(sm.group(1)) if sm else ""
        chapters.append(Chapter(number, subtitle, paragraphs))
    return chapters


def book_title_from(path: Path) -> str:
    stem = path.stem
    stem = re.split(r"_chapters?_", stem)[0]
    return " ".join(w.capitalize() for w in stem.replace("-", "_").split("_"))


# ----------------------------------------------------------------------------- synthesis

class Synth:
    def __init__(self, voice: str, speed: float):
        from mlx_audio.tts.utils import load_model  # heavy import, only when synthesizing

        self.voice = voice
        self.speed = speed
        self.model = load_model(MODEL_ID)
        self.pipeline = self.model._get_pipeline("a")  # American English; created once

    def say(self, text: str) -> np.ndarray:
        """Synthesize one paragraph. The pipeline may split it into several segments."""
        pieces: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in self.pipeline(
            text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
        ):
            if audio is None:
                continue
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            if pieces:
                pieces.append(silence(PAUSE_WITHIN_PARAGRAPH))
            pieces.append(a)
        return np.concatenate(pieces) if pieces else silence(0.2)

    def chapter(self, ch: Chapter) -> np.ndarray:
        parts = [self.say(ch.spoken_title), silence(PAUSE_AFTER_TITLE)]
        for i, p in enumerate(ch.paragraphs):
            if i:
                parts.append(silence(PAUSE_BETWEEN_PARAGRAPHS))
            parts.append(self.say(p))
        parts.append(silence(PAUSE_END_OF_CHAPTER))
        return np.concatenate(parts)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def cache_key(ch: Chapter, voice: str, speed: float) -> str:
    h = hashlib.sha1()
    h.update(f"{MODEL_ID}|{voice}|{speed}|{PIPELINE_VERSION}|".encode())
    h.update(ch.cache_text().encode("utf-8"))
    return h.hexdigest()


def ensure_chapter_audio(ch: Chapter, cache_dir: Path, synth_factory, voice: str, speed: float, log) -> Path:
    """Return the cached wav for this chapter, synthesizing it if missing."""
    key = cache_key(ch, voice, speed)
    wav = cache_dir / f"{key}.wav"
    meta = cache_dir / f"{key}.json"
    if wav.exists() and meta.exists():
        info = json.loads(meta.read_text())
        log(f"ch {ch.number:>5}  {ch.words:>5} words  {fmt(info['duration'])} audio  cached")
        return wav
    synth = synth_factory()
    t0 = time.time()
    audio = synth.chapter(ch)
    gen = time.time() - t0
    duration = len(audio) / SAMPLE_RATE
    tmp = wav.with_suffix(".tmp.wav")
    sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16")
    tmp.replace(wav)
    meta.write_text(json.dumps({
        "number": ch.number, "subtitle": ch.subtitle, "words": ch.words,
        "duration": duration, "generate_seconds": gen, "voice": voice, "speed": speed,
        "model": MODEL_ID, "pipeline_version": PIPELINE_VERSION,
    }, indent=1))
    log(f"ch {ch.number:>5}  {ch.words:>5} words  {fmt(duration)} audio  {gen:6.1f}s gen  {duration / gen:5.1f}x realtime")
    return wav


# ----------------------------------------------------------------------------- assembly

def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def ffmeta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def build_m4b(items: list[tuple[Chapter, Path]], out: Path, book: str, work_dir: Path, log) -> Path:
    """Concatenate chapter wavs into one M4B with chapter markers."""
    first, last = items[0][0].number, items[-1][0].number
    album = f"{book} - Ch {first}-{last}"
    list_file = work_dir / f"{out.stem}.concat.txt"
    meta_file = work_dir / f"{out.stem}.ffmeta"
    lines = [f"file '{p.as_posix()}'" for _, p in items]
    list_file.write_text("\n".join(lines) + "\n")

    meta = [";FFMETADATA1", f"title={ffmeta_escape(album)}", f"album={ffmeta_escape(book)}",
            f"artist={ffmeta_escape(book)}", "genre=Audiobook", ""]
    t = 0.0
    for ch, p in items:
        d = wav_duration(p)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(t * 1000)}", f"END={int((t + d) * 1000)}",
                 f"title={ffmeta_escape(ch.marker_title)}", ""]
        t += d
    meta_file.write_text("\n".join(meta))

    tmp = out.with_suffix(".tmp.m4b")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-i", str(meta_file), "-map_metadata", "1", "-map", "0:a",
           "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    log(f"wrote {out.name}  ({len(items)} chapters, {fmt(t)})")
    return out


def group_chapters(chapters: list[Chapter], group: int) -> list[list[Chapter]]:
    chapters = sorted(chapters, key=lambda c: c.number)
    first = chapters[0].number
    groups: dict[int, list[Chapter]] = {}
    for ch in chapters:
        groups.setdefault((ch.number - first) // group, []).append(ch)
    return [groups[k] for k in sorted(groups)]


# ----------------------------------------------------------------------------- commands

def fmt(seconds: float) -> str:
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def make(txt: Path, out_dir: Path, voice: str, speed: float, group: int, limit: int | None,
         title: str | None, dry_run: bool, log) -> list[Path]:
    txt = txt.resolve()
    chapters = parse_chapters(txt)
    if limit:
        chapters = chapters[:limit]
    book = title or book_title_from(txt)
    total_words = sum(c.words for c in chapters)
    log(f"{txt.name}: {len(chapters)} chapters, {total_words} words, book '{book}', "
        f"{len(group_chapters(chapters, group))} file(s) of up to {group} chapters")
    if dry_run:
        for c in chapters:
            log(f"  ch {c.number:>5}  {c.words:>5} words  {len(c.paragraphs):>3} paras  {c.subtitle}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    synth_holder: dict[str, Synth] = {}

    def synth_factory() -> Synth:
        if "s" not in synth_holder:
            synth_holder["s"] = Synth(voice, speed)
        return synth_holder["s"]

    outputs: list[Path] = []
    for grp in group_chapters(chapters, group):
        first, last = grp[0].number, grp[-1].number
        out = out_dir / f"{book} - Ch {first}-{last}.m4b"
        items = [(c, ensure_chapter_audio(c, cache_dir, synth_factory, voice, speed, log)) for c in grp]
        newest_wav = max(p.stat().st_mtime for _, p in items)
        if out.exists() and out.stat().st_mtime >= newest_wav:
            log(f"up to date: {out.name}")
        else:
            build_m4b(items, out, book, cache_dir, log)
        outputs.append(out)
    return outputs


def queue(folder: Path, archived: str, out_dir: Path | None, **kw) -> list[Path]:
    """Every top-level PDF in `folder` is wanted; its text lives in `folder/archived/`. PDFs moved
    elsewhere (e.g. a Read/ subfolder) are skipped simply because they are no longer top-level."""
    log = kw["log"]
    outputs: list[Path] = []
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem)] or [0])
    if not pdfs:
        log(f"no top-level PDFs in {folder}")
    for pdf in pdfs:
        txt = folder / archived / f"{pdf.stem}.txt"
        if not txt.exists():
            log(f"skip {pdf.name}: no {txt.relative_to(folder)}")
            continue
        outputs += make(txt, out_dir or folder / "audiobooks", **kw)
    return outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--voice", default="af_heart", help="Kokoro voice id (default af_heart)")
        p.add_argument("--speed", type=float, default=1.0)
        p.add_argument("--group", type=int, default=25, help="chapters per M4B (default 25)")
        p.add_argument("--limit", type=int, help="only the first N chapters (for testing)")
        p.add_argument("--title", help="book title (default derived from the filename)")
        p.add_argument("--dry-run", action="store_true", help="parse and list chapters, no audio")
        p.add_argument("--log-file", type=Path, help="append progress lines here as well as stdout")

    pm = sub.add_parser("make", help="one text file -> M4B files")
    pm.add_argument("txt", type=Path)
    pm.add_argument("--out-dir", type=Path, default=Path("audiobooks"))
    common(pm)

    pq = sub.add_parser("queue", help="every top-level PDF in a folder whose text is in archived/")
    pq.add_argument("folder", type=Path)
    pq.add_argument("--archived", default="archived")
    pq.add_argument("--out-dir", type=Path, help="default <folder>/audiobooks")
    common(pq)

    a = ap.parse_args(argv)

    def log(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        if a.log_file:
            with open(a.log_file, "a") as f:
                f.write(f"[{stamp}] {line}\n")

    kw = dict(voice=a.voice, speed=a.speed, group=a.group, limit=a.limit, title=a.title,
              dry_run=a.dry_run, log=log)
    t0 = time.time()
    if a.cmd == "make":
        outs = make(a.txt, a.out_dir, **kw)
    else:
        outs = queue(a.folder, a.archived, a.out_dir, **kw)
    log(f"RESULT: {len(outs)} file(s) in {fmt(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
