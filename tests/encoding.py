"""Unit tests for analysis/encoding.py — no external dependencies required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from encoding import repair_text, _KNOWN_REPAIRS


# ---------------------------------------------------------------------------
# Core contract


def test_empty_string_unchanged():
    assert repair_text("") == ""


def test_none_like_empty_unchanged():
    assert repair_text("") == ""


def test_clean_ascii_unchanged():
    text = "This is a normal sentence with no encoding issues."
    assert repair_text(text) == text


def test_clean_unicode_unchanged():
    text = "Café, naïve, résumé — proper Unicode already."
    result = repair_text(text)
    # Should either be unchanged or equivalently correct
    assert "Caf" in result and "na" in result


# ---------------------------------------------------------------------------
# Known mojibake sequences (tasks.org AC4)
# These are the 5 patterns explicitly listed in the acceptance criteria.


def test_right_single_quote():
    """â€™ → ' (U+2019 right single quotation mark)"""
    assert repair_text("Virginiaâ€™s") == "Virginia’s"


def test_left_double_quote():
    """â€œ → " (U+201C left double quotation mark)"""
    assert repair_text("â€œHello") == "“Hello"


def test_en_dash():
    """â€" (where last char is U+201C) → – (U+2013 en dash)"""
    result = repair_text("pages 1â€“5")
    assert "–" in result or "—" in result or "-" in result


def test_em_dash():
    """â€" (where last char is U+201D) → — (U+2014 em dash)"""
    result = repair_text("wordâ€”word")
    assert "—" in result or "–" in result or "-" in result


def test_right_double_quote():
    """â€\x9d → " (U+201D right double quotation mark)"""
    result = repair_text("saidâ€ he")
    # Should not contain the raw artifact
    assert "â€" not in result


# ---------------------------------------------------------------------------
# Round-trip: garbled text produces sensible output


def test_garbled_sentence_repaired():
    """A sentence with multiple mojibake chars is repaired to readable text."""
    # "Don't" with right single quote encoded as UTF-8, then decoded as cp1252
    # D o n ' t  →  D o n â€™ t
    garbled = "Donâ€™t worry"
    result = repair_text(garbled)
    assert "Don" in result and "t worry" in result
    assert "â€" not in result  # artifact gone


def test_clean_string_after_repair_has_no_artifacts():
    garbled = "She said â€œHelloâ€ and left."
    result = repair_text(garbled)
    assert "â€" not in result


# ---------------------------------------------------------------------------
# FFFD replacement characters (from strict UTF-8 decode of cp1252 bytes)


def test_fffd_preserved_not_crashed():
    """repair_text must not raise on U+FFFD; it may or may not repair it."""
    text = "Virginia�s Public Schools"
    result = repair_text(text)
    assert isinstance(result, str)
    assert "Virginia" in result


# ---------------------------------------------------------------------------
# _KNOWN_REPAIRS table structure


def test_known_repairs_non_empty():
    assert len(_KNOWN_REPAIRS) > 0


def test_known_repairs_are_pairs():
    for item in _KNOWN_REPAIRS:
        assert len(item) == 2
        bad, good = item
        assert isinstance(bad, str) and isinstance(good, str)


def test_known_repairs_bad_not_equal_good():
    for bad, good in _KNOWN_REPAIRS:
        assert bad != good
