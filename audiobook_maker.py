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
the M4B files (one per --group chapters, default 25, or --split sizes) with
chapter markers. --chapters N-M selects a range of one file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from textprep import Lexicon, normalize

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
    paragraphs: list[str]  # the book's text, normalized; what `check` compares the audio against
    lexicon: Lexicon = field(default_factory=Lexicon, repr=False, compare=False)

    @property
    def spoken_title(self) -> str:
        t = f"Chapter {self.number}. {self.subtitle}" if self.subtitle else f"Chapter {self.number}"
        return self.lexicon.apply(t)

    @property
    def spoken_paragraphs(self) -> list[str]:
        """What the narrator is asked to say: the paragraphs with the book's lexicon applied."""
        return [self.lexicon.apply(p) for p in self.paragraphs]

    @property
    def marker_title(self) -> str:
        return f"Chapter {self.number}: {self.subtitle}" if self.subtitle else f"Chapter {self.number}"

    @property
    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)

    def cache_text(self) -> str:
        # With an empty lexicon this is byte-identical to pipeline version 1's key text, so a new
        # lexicon entry (or text rule) re-synthesizes only the chapters whose spoken text it changes.
        return "\n".join([self.spoken_title, *self.spoken_paragraphs])



def parse_chapters(path: Path, lexicon: Lexicon | None = None) -> list[Chapter]:
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
        chapters.append(Chapter(number, subtitle, paragraphs, lexicon or Lexicon()))
    return chapters


def book_title_from(path: Path) -> str:
    stem = path.stem
    stem = re.split(r"_chapters?_", stem)[0]
    return " ".join(w.capitalize() for w in stem.replace("-", "_").split("_"))


# ----------------------------------------------------------------------------- synthesis

def parse_voice(spec: str) -> list[tuple[str, float]]:
    """"am_liam", or a weighted blend "am_liam:0.7,am_michael:0.3" (weights are normalized)."""
    parts = []
    for item in spec.split(","):
        name, _, w = item.strip().partition(":")
        if not re.fullmatch(r"[a-z]{2}_[a-z0-9]+", name):
            raise SystemExit(f"voice {name!r} in {spec!r} is not a Kokoro voice id like am_liam")
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            raise SystemExit(f"weight {w!r} in {spec!r} is not a number")
        if weight <= 0:
            raise SystemExit(f"weight for {name} in {spec!r} must be positive")
        parts.append((name, weight))
    return parts


class Synth:
    def __init__(self, voice: str, speed: float, blend_dir: Path = Path(".cache")):
        from mlx_audio.tts.utils import load_model  # heavy import, only when synthesizing

        self.speed = speed
        self.model = load_model(MODEL_ID)
        self.pipeline = self.model._get_pipeline("a")  # American English; created once
        self.voice = self._resolve_voice(voice, blend_dir)

    def _resolve_voice(self, spec: str, blend_dir: Path) -> str:
        """A single voice id passes through. A blend is the weighted mean of the voice tensors
        (voices are style vectors, so mixing needs no training), saved once as a .safetensors file
        that the pipeline loads like any voice. Cache keys use the spec string, not the file."""
        parts = parse_voice(spec)
        if len(parts) == 1:
            return parts[0][0]
        import mlx.core as mx

        total = sum(w for _, w in parts)
        name = "+".join(f"{n}-{w / total:.3f}" for n, w in parts)
        path = blend_dir / "voices" / f"{name}.safetensors"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            mixed = sum(self.pipeline.load_single_voice(n) * (w / total) for n, w in parts)
            mx.save_safetensors(str(path), {"voice": mixed})
        return str(path)

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
        for i, p in enumerate(ch.spoken_paragraphs):
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


def group_chapters(chapters: list[Chapter], group: int, split: list[int] | None = None,
                   merge_tail: bool = False) -> list[list[Chapter]]:
    """Consecutive runs of `group` chapters. `split` gives explicit sizes instead (they must add up
    to the chapter count); `merge_tail` folds a short last group into the one before it."""
    chapters = sorted(chapters, key=lambda c: c.number)
    if split:
        if sum(split) != len(chapters):
            raise SystemExit(f"--split {','.join(map(str, split))} adds up to {sum(split)}, "
                             f"but {len(chapters)} chapters are selected")
        out, i = [], 0
        for n in split:
            out.append(chapters[i:i + n])
            i += n
        return out
    first = chapters[0].number
    groups: dict[int, list[Chapter]] = {}
    for ch in chapters:
        groups.setdefault((ch.number - first) // group, []).append(ch)
    out = [groups[k] for k in sorted(groups)]
    if merge_tail and len(out) > 1 and len(out[-1]) < group:
        tail = out.pop()  # pop first: `out[-2] += out.pop()` stores into the shifted slot
        out[-1] = out[-1] + tail
    return out


def parse_range(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"expected N or N-M, got {s!r}")
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    if hi < lo:
        raise argparse.ArgumentTypeError(f"range {s!r} runs backwards")
    return lo, hi


def parse_split(s: str) -> list[int]:
    try:
        sizes = [int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected sizes like 10,10,12, got {s!r}")
    if any(n < 1 for n in sizes):
        raise argparse.ArgumentTypeError("every --split size must be at least 1")
    return sizes


def seconds_per_word(cache_dir: Path) -> float:
    """Measured synthesis cost from the cache's own metadata; a first-run guess when it is empty."""
    words = secs = 0.0
    for meta in cache_dir.glob("*.json"):
        try:
            info = json.loads(meta.read_text())
            words += info["words"]
            secs += info["generate_seconds"]
        except (ValueError, KeyError):
            continue
    return secs / words if words else 0.045


# ----------------------------------------------------------------------------- phase 2: find problems

def select_chapters(txt: Path, chapters_range: tuple[int, int] | None, lexicon: Lexicon) -> list[Chapter]:
    chapters = parse_chapters(txt.resolve(), lexicon)
    if chapters_range:
        lo, hi = chapters_range
        chapters = [c for c in chapters if lo <= c.number <= hi]
        if not chapters:
            raise SystemExit(f"{txt.name} has no chapters in {lo}-{hi}")
    return chapters


def oov(txt: Path, chapters_range, lexicon_path: Path | None, top: int, log) -> list[tuple[str, int]]:
    """Words misaki has no dictionary entry for, so espeak guesses them — the lexicon's candidates.
    Measured by watching which words reach the fallback while phonemizing the real paragraphs."""
    from misaki import en, espeak

    lexicon = Lexicon.find(txt.resolve(), lexicon_path)
    chapters = select_chapters(txt, chapters_range, lexicon)
    g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
    inner, seen = g2p.fallback, {}

    def spy(tk):
        w = re.sub(r"['’]s$", "", tk.text.strip("…—-.,;:!?\"“”‘’()[]"))
        if re.search(r"[A-Za-z]", w):
            seen[w] = seen.get(w, 0) + 1
        return inner(tk)

    g2p.fallback = spy
    for c in chapters:
        for para in [c.spoken_title, *c.spoken_paragraphs]:
            g2p(para)
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    log(f"{len(chapters)} chapters: {len(ranked)} words go to the espeak fallback"
        + (f" (lexicon {lexicon.source} already covers the rest)" if lexicon else ""))
    log("paste the ones that sound wrong into lexicon.json and fix the phonemes (or use a respelling):")
    g2p.fallback = inner
    for w, n in ranked[:top]:
        print(f'  "{w}": "/{g2p(w)[0]}/",   # {n}x')
    return ranked


def find_cached_wav(cache_dir: Path, number: int, voice: str | None) -> tuple[Path, dict] | None:
    """Newest cached render of chapter `number` (in `voice`, when given)."""
    best = None
    for meta in cache_dir.glob("*.json"):
        try:
            info = json.loads(meta.read_text())
        except ValueError:
            continue
        if info.get("number") != number or (voice and info.get("voice") != voice):
            continue
        wav = meta.with_suffix(".wav")
        if wav.exists() and (best is None or wav.stat().st_mtime > best[0].stat().st_mtime):
            best = (wav, info)
    return best


_NUM = re.compile(r"^\d+$")


def words_for_diff(text: str) -> list[str]:
    """Lower-case words with punctuation dropped; digits spelled out (Whisper writes "1262")."""
    from num2words import num2words

    out: list[str] = []
    for w in re.split(r"[\s—–-]+", text.replace("’", "'")):
        w = re.sub(r"[^\w']", "", w).strip("'").lower()
        if not w:
            continue
        if _NUM.match(w):
            out += re.split(r"[\s-]+", num2words(int(w)).replace(",", "").replace(" and ", " "))
        else:
            out.append(w)
    return out


def check(txt: Path, chapters_range, out_dir: Path, voice: str | None, model: str,
          lexicon_path: Path | None, log) -> dict[str, int]:
    """Whisper round trip: transcribe each cached chapter and diff it against the source text.
    A word Whisper hears as something else is either mispronounced or a name Whisper cannot
    spell; the report ranks them across chapters so the lexicon work starts from the worst."""
    import difflib

    import mlx_whisper

    lexicon = Lexicon.find(txt.resolve(), lexicon_path)
    chapters = select_chapters(txt, chapters_range, lexicon)
    cache_dir, report_dir = out_dir / ".cache", out_dir / "check"
    report_dir.mkdir(parents=True, exist_ok=True)
    suspects: dict[str, int] = {}
    for c in chapters:
        found = find_cached_wav(cache_dir, c.number, voice)
        if not found:
            log(f"ch {c.number:>5}  no cached audio{' in ' + voice if voice else ''}; skipped")
            continue
        wav, info = found
        # Names in the prompt let Whisper spell them, so a correctly spoken name is not flagged.
        names = sorted({w for p in c.paragraphs for w in re.findall(r"\b[A-Z][a-z]{3,}\b", p)})
        prompt = ", ".join([*lexicon.entries, *names])[:800]
        t0 = time.time()
        res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, word_timestamps=True,
                                     initial_prompt=prompt, condition_on_previous_text=False)
        heard = [(w["word"], w["start"]) for seg in res["segments"] for w in seg.get("words", [])]
        hyp, hyp_t = [], []
        for raw, start in heard:
            for w in words_for_diff(raw):
                hyp.append(w)
                hyp_t.append(start)
        src = words_for_diff("\n".join([f"Chapter {c.number}. {c.subtitle}", *c.paragraphs]))
        sm = difflib.SequenceMatcher(None, src, hyp, autojunk=False)
        issues, edits = [], 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            edits += max(i2 - i1, j2 - j1)
            at = hyp_t[j1] if j1 < len(hyp_t) else (hyp_t[-1] if hyp_t else 0.0)
            said, got = " ".join(src[i1:i2]), " ".join(hyp[j1:j2])
            issues.append((at, op, said, got))
            for w in src[i1:i2]:
                suspects[w] = suspects.get(w, 0) + 1
        wer = edits / max(1, len(src))
        lines = [f"# Chapter {c.number} — Whisper round trip", "",
                 f"voice {info.get('voice')}, {fmt(info.get('duration', 0))}, {len(src)} words, "
                 f"{len(issues)} differences, word error rate {wer:.1%}", "",
                 "| at | text says | Whisper heard |", "|---|---|---|"]
        lines += [f"| {fmt(at)} | {said or '—'} | {got or '—'} |" for at, op, said, got in issues]
        (report_dir / f"ch-{c.number}.md").write_text("\n".join(lines) + "\n")
        log(f"ch {c.number:>5}  {len(src):>5} words  {len(issues):>3} diffs  WER {wer:5.1%}  "
            f"{time.time() - t0:5.1f}s  -> check/ch-{c.number}.md")
    ranked = sorted(suspects.items(), key=lambda kv: -kv[1])
    common = {"the", "a", "and", "of", "to", "was", "he", "his", "that", "it", "in", "had", "as"}
    top = [(w, n) for w, n in ranked if w not in common][:25]
    if top:
        log("most-missed source words (lexicon candidates): " + ", ".join(f"{w} {n}x" for w, n in top))
    return suspects


# ----------------------------------------------------------------------------- phase 2: voices

def audition(voices: list[str], txt: Path | None, chapter: int | None, paragraphs: int, text: str | None,
             out_dir: Path, speed: float, lexicon_path: Path | None, log) -> list[Path]:
    """Render the same passage in each voice or blend: one m4a per voice plus one file of all of
    them in order, each clip introduced by its number ("Voice two.") so the ear can match it up."""
    from num2words import num2words

    for v in voices:
        parse_voice(v)
    if text:
        passage = [normalize(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
        lexicon = Lexicon.find(txt.resolve(), lexicon_path) if txt else Lexicon.load(lexicon_path)
    else:
        if not txt or chapter is None:
            raise SystemExit("audition needs --text, or a text file with --chapter N")
        lexicon = Lexicon.find(txt.resolve(), lexicon_path)
        chs = select_chapters(txt, (chapter, chapter), lexicon)
        passage = chs[0].paragraphs[:paragraphs]
    passage = [lexicon.apply(p) for p in passage]
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".work"
    work.mkdir(exist_ok=True)
    synth: Synth | None = None
    clips: list[Path] = []
    for i, spec in enumerate(voices, 1):
        if synth is None:
            synth = Synth(spec, speed, work)
        else:
            synth.voice = synth._resolve_voice(spec, work)
        parts = [synth.say(f"Voice {num2words(i)}."), silence(0.8)]
        for j, p in enumerate(passage):
            if j:
                parts.append(silence(PAUSE_BETWEEN_PARAGRAPHS))
            parts.append(synth.say(p))
        parts.append(silence(1.2))
        audio = np.concatenate(parts)
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", spec.replace(":", "-").replace(",", "+"))
        wav = work / f"{i:02d}-{safe}.wav"
        sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        clips.append(wav)
        m4a = out_dir / f"{i:02d}-{safe}.m4a"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "64k",
                        "-ac", "1", str(m4a)], check=True)
        log(f"voice {i:>2}  {spec:<32} {fmt(len(audio) / SAMPLE_RATE)}  -> {m4a.name}")
    list_file = work / "all.concat.txt"
    list_file.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    everything = out_dir / "00-all-voices.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c:a", "aac", "-b:a", "64k", "-ac", "1", str(everything)], check=True)
    log(f"wrote {everything}")
    return [everything]

# ----------------------------------------------------------------------------- commands

def fmt(seconds: float) -> str:
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def make(txt: Path, out_dir: Path, voice: str, speed: float, group: int, limit: int | None,
         title: str | None, dry_run: bool, log, chapters_range: tuple[int, int] | None = None,
         split: list[int] | None = None, merge_tail: bool = False,
         lexicon_path: Path | None = None) -> list[Path]:
    txt = txt.resolve()
    parse_voice(voice)  # fail on a bad spec before anything slow
    lexicon = Lexicon.find(txt, lexicon_path)
    chapters = parse_chapters(txt, lexicon)
    if chapters_range:
        lo, hi = chapters_range
        chapters = [c for c in chapters if lo <= c.number <= hi]
        missing = sorted(set(range(lo, hi + 1)) - {c.number for c in chapters})
        if missing:
            raise SystemExit(f"{txt.name} has no chapter(s) {missing[:10]} in {lo}-{hi}")
    if limit:
        chapters = chapters[:limit]
    book = title or book_title_from(txt)
    total_words = sum(c.words for c in chapters)
    groups = group_chapters(chapters, group, split, merge_tail)
    log(f"{txt.name}: {len(chapters)} chapters, {total_words} words, book '{book}', "
        f"{len(groups)} file(s): " + ", ".join(f"{g[0].number}-{g[-1].number}" for g in groups))

    cache_dir = out_dir / ".cache"
    new = [c for c in chapters if not (cache_dir / f"{cache_key(c, voice, speed)}.wav").exists()]
    est = sum(c.words for c in new) * seconds_per_word(cache_dir)
    if lexicon:
        touched = sum(1 for c in chapters if lexicon.hits("\n".join([c.subtitle, *c.paragraphs])))
        log(f"lexicon: {len(lexicon.entries)} entries from {lexicon.source}, "
            f"used in {touched} of {len(chapters)} chapters")
    log(f"plan: {len(chapters)} chapters ({len(new)} new, {len(chapters) - len(new)} cached), "
        f"{len(groups)} file(s), est ~{fmt(est)}")
    if dry_run:
        for c in chapters:
            state = "new" if c in new else "cached"
            log(f"  ch {c.number:>5}  {c.words:>5} words  {len(c.paragraphs):>3} paras  {state:<6}  {c.subtitle}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    synth_holder: dict[str, Synth] = {}

    def synth_factory() -> Synth:
        if "s" not in synth_holder:
            synth_holder["s"] = Synth(voice, speed, cache_dir)
        return synth_holder["s"]

    outputs: list[Path] = []
    for grp in groups:
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
        p.add_argument("--voice", default="am_liam",
                       help="Kokoro voice id, or a blend like am_liam:0.7,am_michael:0.3 (default am_liam)")
        p.add_argument("--speed", type=float, default=1.0)
        p.add_argument("--group", type=int, default=25, help="chapters per M4B (default 25)")
        p.add_argument("--merge-tail", action="store_true",
                       help="fold a short last group into the one before it (52 by 10 -> 10,10,10,10,12)")
        p.add_argument("--limit", type=int, help="only the first N chapters (for testing)")
        p.add_argument("--title", help="book title (default derived from the filename)")
        p.add_argument("--dry-run", action="store_true", help="parse and list chapters, no audio")
        p.add_argument("--log-file", type=Path, help="append progress lines here as well as stdout")
        p.add_argument("--lexicon", type=Path,
                       help="pronunciation lexicon (default lexicon.json beside the text file or one folder up)")

    pm = sub.add_parser("make", help="one text file -> M4B files")
    pm.add_argument("txt", type=Path)
    pm.add_argument("--out-dir", type=Path, default=Path("audiobooks"))
    pm.add_argument("--chapters", type=parse_range, help="only chapters N-M (inclusive)")
    pm.add_argument("--split", type=parse_split, help="explicit chapters per file, e.g. 10,10,10,10,12")
    common(pm)

    pq = sub.add_parser("queue", help="every top-level PDF in a folder whose text is in archived/")
    pq.add_argument("folder", type=Path)
    pq.add_argument("--archived", default="archived")
    pq.add_argument("--out-dir", type=Path, help="default <folder>/audiobooks")
    common(pq)

    po = sub.add_parser("oov", help="words Kokoro has to guess (espeak fallback): lexicon candidates")
    po.add_argument("txt", type=Path)
    po.add_argument("--chapters", type=parse_range)
    po.add_argument("--top", type=int, default=40)
    po.add_argument("--lexicon", type=Path)

    pc = sub.add_parser("check", help="Whisper round trip: transcribe cached chapters, diff against the text")
    pc.add_argument("txt", type=Path)
    pc.add_argument("--chapters", type=parse_range, required=True)
    pc.add_argument("--out-dir", type=Path, default=Path("audiobooks"), help="where .cache/ is; reports go to check/")
    pc.add_argument("--voice", help="only renders in this voice (default: newest render of each chapter)")
    pc.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    pc.add_argument("--lexicon", type=Path)

    pa = sub.add_parser("audition", help="the same passage in several voices or blends (am_liam:0.7,am_michael:0.3)")
    pa.add_argument("voices", nargs="+")
    pa.add_argument("--txt", type=Path, help="text file to take the passage from (with --chapter)")
    pa.add_argument("--chapter", type=int)
    pa.add_argument("--paragraphs", type=int, default=4)
    pa.add_argument("--text", help="or the passage itself")
    pa.add_argument("--out-dir", type=Path, default=Path("audiobooks/compare/auditions"))
    pa.add_argument("--speed", type=float, default=1.0)
    pa.add_argument("--lexicon", type=Path)

    a = ap.parse_args(argv)

    def log(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        if getattr(a, "log_file", None):
            with open(a.log_file, "a") as f:
                f.write(f"[{stamp}] {line}\n")

    t0 = time.time()
    if a.cmd == "oov":
        oov(a.txt, a.chapters, a.lexicon, a.top, log)
        return 0
    if a.cmd == "check":
        check(a.txt, a.chapters, a.out_dir, a.voice, a.model, a.lexicon, log)
        log(f"RESULT: checked in {fmt(time.time() - t0)}")
        return 0
    if a.cmd == "audition":
        outs = audition(a.voices, a.txt, a.chapter, a.paragraphs, a.text, a.out_dir, a.speed, a.lexicon, log)
        log(f"RESULT: {len(outs)} file(s) in {fmt(time.time() - t0)}")
        return 0
    kw = dict(voice=a.voice, speed=a.speed, group=a.group, limit=a.limit, title=a.title,
              dry_run=a.dry_run, log=log, merge_tail=a.merge_tail, lexicon_path=a.lexicon)
    t0 = time.time()
    if a.cmd == "make":
        outs = make(a.txt, a.out_dir, chapters_range=a.chapters, split=a.split, **kw)
    else:
        outs = queue(a.folder, a.archived, a.out_dir, **kw)
    log(f"RESULT: {len(outs)} file(s) in {fmt(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
