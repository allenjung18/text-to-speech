"""Golden tests: input text -> what the narrator is asked to say. Run with `uv run pytest`.

A failing case here means a change to the text rules alters spoken text, and therefore re-synthesizes
every chapter it touches. That can be the point of the change; update the expected value on purpose.
"""
import hashlib
import json

import pytest

from audiobook_maker import Chapter, cache_key, parse_chapters, parse_voice, words_for_diff
from textprep import Lexicon, normalize, possessive

NORMALIZE = [
    # unchanged since pipeline version 1
    ("plain sentence.", "plain sentence."),
    ("a b​c﻿", "a bc"),
    ("wait--what", "wait—what"),
    ("too   many \t spaces", "too many spaces"),
    ("  trimmed  ", "trimmed"),
    ("guilt-ridden armor-type", "guilt-ridden armor-type"),  # compound hyphens are words, not dashes
    ("Huh… what?", "Huh… what?"),
    # spaced hyphen is a dash: misaki drops " - " silently, the em dash gets a pause
    ("It was - bad", "It was — bad"),
    ("- leading dash stays", "- leading dash stays"),
    # system messages lose their brackets ("[" and ")" otherwise leak into the phonemes)
    ("[You have slain a Corrupted Beast, River Locust.]", "You have slain a Corrupted Beast, River Locust."),
    ("[Fated] and [Your shadow grows stronger.]", "Fated and Your shadow grows stronger."),
    # inline phoneme overrides are markup, not system messages
    ("[Ananke](/ənˈæŋki/) smiled", "[Ananke](/ənˈæŋki/) smiled"),
]


@pytest.mark.parametrize("raw,spoken", NORMALIZE)
def test_normalize(raw, spoken):
    assert normalize(raw) == spoken


@pytest.mark.parametrize("raw,_", NORMALIZE)
def test_normalize_is_idempotent(raw, _):
    assert normalize(normalize(raw)) == normalize(raw)


LEX = Lexicon({"Ananke": "/ənˈæŋki/", "Nephis": "/nˈɛfɪs/", "Neph": "/nˈɛf/", "NQSC": "N Q S C"})

LEXICON = [
    ("Ananke smiled.", "[Ananke](/ənˈæŋki/) smiled."),
    ("Ananke's boat", "[Ananke's](/ənˈæŋkiz/) boat"),
    ("Ananke’s boat", "[Ananke’s](/ənˈæŋkiz/) boat"),
    ("Nephis's sword", "[Nephis's](/nˈɛfɪsᵻz/) sword"),
    ("Nephis' sword", "[Nephis'](/nˈɛfɪs/) sword"),
    ("Neph and Nephis", "[Neph](/nˈɛf/) and [Nephis](/nˈɛfɪs/)"),  # longest key first, whole words
    ("Nephilim", "Nephilim"),
    ("back to NQSC?", "back to N Q S C?"),
    ("ananke", "ananke"),  # case-sensitive
    ("[Ananke](/x/) already marked", "[Ananke](/x/) already marked"),
]


@pytest.mark.parametrize("raw,spoken", LEXICON)
def test_lexicon(raw, spoken):
    assert LEX.apply(raw) == spoken


def test_possessive_sound():
    assert possessive("nˈɛf") == "nˈɛfs"
    assert possessive("nˈɛfɪs") == "nˈɛfɪsᵻz"
    assert possessive("ənˈæŋki") == "ənˈæŋkiz"


def test_empty_lexicon_keeps_cache_key(tmp_path):
    """The chapters already on disk were keyed as sha1(model|voice|speed|1| + title + paragraphs).
    Without a lexicon entry that matches, the key must not move, or every finished chapter
    would be synthesized again."""
    ch = Chapter(1262, "A Shrimp Between Two Whales", ["Sunny described it.", "It was clear."])
    old = hashlib.sha1()
    old.update(b"mlx-community/Kokoro-82M-bf16|am_liam|1.0|1|")
    old.update("Chapter 1262. A Shrimp Between Two Whales\nSunny described it.\nIt was clear.".encode())
    assert cache_key(ch, "am_liam", 1.0) == old.hexdigest()
    ch.lexicon = Lexicon({"Nephis": "/nˈɛfɪs/"})  # no match in this chapter
    assert cache_key(ch, "am_liam", 1.0) == old.hexdigest()
    ch.lexicon = Lexicon({"Sunny": "/sˈʌni/"})
    assert cache_key(ch, "am_liam", 1.0) != old.hexdigest()


def test_parse_chapters_with_lexicon(tmp_path):
    book = tmp_path / "shadow_slave_chapters_1_2.txt"
    book.write_text(
        "Shadow Slave Chapter 1 - Chapter 1 - 1: Ananke\n\nFirst para.\n\n[You got a Memory.]\n"
        + "-" * 40 + "\nShadow Slave Chapter 2 - Chapter 2 - 2: Two\n\nNephis - again.\n"
    )
    (tmp_path / "lexicon.json").write_text(json.dumps({"_comment": "x", "Ananke": "/ənˈæŋki/"}))
    lex = Lexicon.find(book)
    ch = parse_chapters(book, lex)
    assert [c.number for c in ch] == [1, 2]
    assert ch[0].spoken_title == "Chapter 1. [Ananke](/ənˈæŋki/)"
    assert ch[0].paragraphs == ["First para.", "You got a Memory."]
    assert ch[1].paragraphs == ["Nephis — again."]


def test_lexicon_rejects_bad_entries(tmp_path):
    f = tmp_path / "lexicon.json"
    f.write_text(json.dumps({"two words": "x"}))
    with pytest.raises(SystemExit):
        Lexicon.load(f)


def test_parse_voice():
    assert parse_voice("am_liam") == [("am_liam", 1.0)]
    assert parse_voice("am_liam:0.7,am_michael:0.3") == [("am_liam", 0.7), ("am_michael", 0.3)]
    for bad in ("Liam", "am_liam:0", "am_liam:x"):
        with pytest.raises(SystemExit):
            parse_voice(bad)


def test_words_for_diff():
    assert words_for_diff("Chapter 1262. “Huh… what’s—that?”") == [
        "chapter", "one", "thousand", "two", "hundred", "sixty", "two", "huh", "what's", "that"]
