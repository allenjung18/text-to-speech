"""textprep — what the narrator is asked to say: text rules and the per-book pronunciation lexicon.

Two layers, applied in this order:

1. `normalize` — conservative text rules every chapter goes through (pinned by
   tests/test_textprep.py). A rule change alters the spoken text, and therefore the cache key, of
   only the chapters it actually touches.
2. `Lexicon.apply` — the book's `lexicon.json`, the user's final say. A value is either a
   respelling ("NQSC": "N Q S C") or misaki phonemes between slashes ("Ananke": "/ənˈæŋki/"),
   which become an inline override `[Ananke](/ənˈæŋki/)`. Keys match whole words, case-sensitive,
   longest first, and also match the possessive (`Ananke's` gets the phonemes plus the right
   possessive sound). A chapter containing none of the keys is left byte-identical, so adding an
   entry re-synthesizes only the chapters that contain that word.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LEXICON_NAME = "lexicon.json"


def normalize(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")  # nbsp, zero-width, BOM
    s = s.replace("--", "—")  # ASCII double dash -> em dash (Kokoro reads it as a pause)
    # A hyphen with a space on both sides is a dash; misaki drops it silently, the em dash pauses.
    s = re.sub(r"(?<=\S) +- +(?=\S)", " — ", s)
    # System messages — "[You have slain a Corrupted Beast.]" — lose their brackets: misaki glues
    # "[" onto the first word (sending it to the espeak fallback) and speaks a stray ")" symbol.
    # Inline phoneme overrides "[word](/…/)" are left alone.
    s = re.sub(r"\[([^\[\]]+)\](?!\()", r"\1", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# The sound of a possessive 's after a word's last phoneme (misaki's own `_s` rule).
_SIBILANTS = set("szʃʒʧʤ")
_VOICELESS = set("ptkfθ")


def possessive(phonemes: str) -> str:
    last = phonemes.rstrip("ˈˌ")[-1:]
    if last in _SIBILANTS:
        return phonemes + "ᵻz"
    if last in _VOICELESS:
        return phonemes + "s"
    return phonemes + "z"


@dataclass
class Lexicon:
    entries: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: Path | None) -> "Lexicon":
        if not path or not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = {k: v for k, v in raw.items() if not k.startswith("_")}  # "_comment" keys allowed
        bad = [k for k, v in entries.items() if not isinstance(v, str) or not v.strip() or " " in k.strip()]
        if bad:
            raise SystemExit(f"{path}: entries must map one word to a non-empty string: {bad[:5]}")
        return cls(entries, path)

    @classmethod
    def find(cls, txt: Path, explicit: Path | None = None) -> "Lexicon":
        """--lexicon if given; else lexicon.json beside the text file or one folder up (the book folder)."""
        if explicit:
            if not explicit.exists():
                raise SystemExit(f"--lexicon {explicit} does not exist")
            return cls.load(explicit)
        for folder in (txt.parent, txt.parent.parent):
            if (folder / LEXICON_NAME).exists():
                return cls.load(folder / LEXICON_NAME)
        return cls()

    def __bool__(self) -> bool:
        return bool(self.entries)

    def _pattern(self) -> re.Pattern | None:
        if not self.entries:
            return None
        keys = sorted(self.entries, key=len, reverse=True)
        alt = "|".join(re.escape(k) for k in keys)
        # possessive: word's / word’s / word' (the last only after s, "Nephis'")
        return re.compile(rf"(?<![\w\[/])({alt})(['’]s\b|(?<=s)['’](?!\w))?(?![\w\]])")

    def apply(self, text: str) -> str:
        pat = self._pattern()
        if pat is None:
            return text

        def sub(m: re.Match) -> str:
            word, poss = m.group(1), m.group(2) or ""
            value = self.entries[word]
            if value.startswith("/") and value.endswith("/") and len(value) > 2:
                ph = value[1:-1]
                if poss:
                    ph = possessive(ph) if poss[-1:] == "s" else ph
                return f"[{word}{poss}](/{ph}/)"
            return value + poss

        return pat.sub(sub, text)

    def hits(self, text: str) -> dict[str, int]:
        pat = self._pattern()
        out: dict[str, int] = {}
        if pat:
            for m in pat.finditer(text):
                out[m.group(1)] = out.get(m.group(1), 0) + 1
        return out
