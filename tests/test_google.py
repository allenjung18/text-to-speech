"""Google Cloud TTS helpers: request splitting, voice tiers, the usage ledger and the budget cut.
No network: make() stops at --dry-run, before any request."""
from pathlib import Path

import audiobook_maker as am
from audiobook_maker import GoogleUsage, google_tier, is_google_voice, split_for_request


def test_voice_detection_and_tier():
    assert is_google_voice("en-US-Chirp3-HD-Algenib")
    assert is_google_voice("en-US-Wavenet-D")
    assert not is_google_voice("am_liam")
    assert not is_google_voice("am_liam:0.7,am_michael:0.3")
    assert google_tier("en-US-Chirp3-HD-Algenib") == "Chirp3-HD"
    assert google_tier("en-US-Wavenet-D") == "Wavenet"


def test_short_text_is_one_request():
    assert split_for_request("Sunny sighed.") == ["Sunny sighed."]


def test_long_text_splits_at_sentences_without_losing_words():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    parts = split_for_request(text, limit=1000)
    assert len(parts) > 1
    assert all(len(p.encode()) <= 1000 for p in parts)
    assert " ".join(parts).split() == text.split()
    assert all(p.endswith(".") for p in parts)


def test_one_sentence_longer_than_a_request_is_cut_on_spaces():
    text = "word " * 1000
    parts = split_for_request(text.strip(), limit=1000)
    assert all(len(p.encode()) <= 1000 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_usage_ledger_counts_per_month_and_tier(tmp_path):
    ledger = tmp_path / "usage.json"
    chirp = GoogleUsage("en-US-Chirp3-HD-Algenib", ledger)
    chirp.add(100)
    chirp.add(50)
    GoogleUsage("en-US-Wavenet-D", ledger).add(7)
    assert chirp.used() == 150
    assert GoogleUsage("en-US-Wavenet-D", ledger).used() == 7


def _book(tmp_path: Path, chapters: int, chars: int) -> Path:
    body = ("x" * (chars - 1) + ".")
    text = "\n" + ("-" * 40 + "\n").join(f"Chapter {n}\n\n{body}\n\n" for n in range(1, chapters + 1))
    txt = tmp_path / "book_chapters_1_9.txt"
    txt.write_text(text)
    return txt


def test_budget_keeps_the_prefix_that_fits(tmp_path, monkeypatch):
    txt = _book(tmp_path, chapters=5, chars=1000)
    monkeypatch.setattr(am, "GOOGLE_USAGE", tmp_path / "usage.json")
    monkeypatch.setattr(am, "GOOGLE_FREE_CHARS", 2500)
    monkeypatch.setattr(GoogleUsage.__init__, "__defaults__", (tmp_path / "usage.json",))
    lines = []
    am.make(txt, tmp_path / "out", voice="en-US-Chirp3-HD-Algenib", speed=1.0, group=100, limit=None,
            title="Book", dry_run=True, log=lines.append)
    plan = next(line for line in lines if line.startswith("plan:"))
    assert plan.startswith("plan: 2 chapters")  # 2 x ~1,011 chars fit in 2,500; the third does not
    assert any("stopping after ch 2" in line for line in lines)
