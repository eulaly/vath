"""
analysis/encoding.py — text encoding repair for scraped content.

The townhall.virginia.gov scraper forces UTF-8 decoding, which is correct for the
site's current content. This module provides a defensive repair function for cases
where a response arrives with Windows-1252/cp1252 bytes embedded in otherwise UTF-8
content (common in older CMSes). The raw scrape files are never modified; repair is
applied at the analysis and reporting layers only.

Primary: uses `ftfy` when installed (pip install ftfy).
Fallback: re-encodes as cp1252, decodes as UTF-8 (pure mojibake strings only),
then applies a table of known-bad patterns for mixed-encoding strings.
"""

# ---------------------------------------------------------------------------
# Known patterns: UTF-8 bytes decoded as cp1252, i.e. the 3-char sequences you
# see when a server sends e.g. E2 80 99 and it gets decoded as cp1252 chars.
#
# Byte → cp1252 char mappings for the 0x80–0x9F range:
#   E2 → â  (U+00E2, always)
#   80 → €  (U+20AC, cp1252 0x80)
#   99 → ™  (U+2122, cp1252 0x99)  ← E2 80 99 = U+2019 ' right single quote
#   98 → ˜  (U+02DC, cp1252 0x98)  ← E2 80 98 = U+2018 ' left single quote
#   9C → œ  (U+0153, cp1252 0x9C)  ← E2 80 9C = U+201C " left double quote
#   9D → \x9d (undefined → U+009D) ← E2 80 9D = U+201D " right double quote
#   93 → "  (U+201C, cp1252 0x93)  ← E2 80 93 = U+2013 – en dash
#   94 → "  (U+201D, cp1252 0x94)  ← E2 80 94 = U+2014 — em dash
#   A6 → ¦  (U+00A6, cp1252 0xA6)  ← E2 80 A6 = U+2026 … ellipsis

_KNOWN_REPAIRS: list[tuple[str, str]] = [
    # Longer / more specific patterns first to avoid partial matches
    ("â€™",  "’"),  # â€™ → ' right single quote
    ("â€˜",  "‘"),  # â€˜ → ' left single quote
    ("â€œ",  "“"),  # â€œ → " left double quote
    ("â€",  "”"),  # â€\x9d → " right double quote
    ("â€“",  "–"),  # â€" (with left DQ) → – en dash
    ("â€”",  "—"),  # â€" (with right DQ) → — em dash
    ("â€¦",  "…"),  # â€¦ → … ellipsis
    # Generic fallback: bare â€ prefix not caught above → remove artifact
    ("â€",        ""),
]


def repair_text(text: str) -> str:
    """Repair common encoding artifacts in scraped text.

    Handles:
    - UTF-8 bytes decoded as cp1252/Latin-1 (â€™ → ')
    - Attempts best-effort cleanup for mixed-encoding strings

    U+FFFD replacement characters (from strict UTF-8 decoding of cp1252 bytes)
    cannot be recovered since the original byte is lost; they are left as-is.
    """
    if not text:
        return text

    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        pass

    # Fallback 1: pure mojibake — entire string is UTF-8 bytes read as cp1252.
    # Re-encode as cp1252 and decode as UTF-8.
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Fallback 2: mixed strings — substitute known-bad patterns.
    for bad, good in _KNOWN_REPAIRS:
        if bad in text:
            text = text.replace(bad, good)
    return text
