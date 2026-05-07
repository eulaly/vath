"""Unit tests for analysis/openai_realtime.py — no real API calls."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
import openai_realtime as rt


# ---------------------------------------------------------------------------
# Fixtures

FORUM_ITEM = {
    "forum_id": "452",
    "reg_title": "Model Policies for Transgender Students",
    "reg_desc": "Guidance developed in response to HB 145.",
}

COMMENT_ITEM = {
    "forum_id": "452",
    "comment_id": "87914",
    "author": "Alice Example",
    "date": "2021-01-04T09:15:00",
    "title": "I support this policy",
    "text": "This is a great policy that protects students.",
}

MOCK_RESPONSE_CONTENT = json.dumps({
    "stance": "support",
    "stance_confidence": 0.95,
    "stance_rationale": "Commenter explicitly endorses the policy.",
    "tone": "positive",
    "tags": ["student safety", "LGBTQ+ inclusion"],
})


def _mock_client(response_content: str = MOCK_RESPONSE_CONTENT):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_content
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# Prompt versioning

def test_prompt_version_is_7_hex_chars():
    assert len(rt.PROMPT_VERSION) == 7
    assert all(c in "0123456789abcdef" for c in rt.PROMPT_VERSION)


def test_prompt_version_matches_prompt_file():
    import hashlib
    prompt_file = Path(__file__).parent.parent / "analysis" / "prompt-1.txt"
    expected = hashlib.sha256(prompt_file.read_text(encoding="utf-8").strip().encode()).hexdigest()[:7]
    assert rt.PROMPT_VERSION == expected


def test_prompt_version_is_stable():
    import hashlib
    v2 = hashlib.sha256(rt.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:7]
    assert v2 == rt.PROMPT_VERSION


# ---------------------------------------------------------------------------
# load_items

def test_load_items_separates_forum_and_comments(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        json.dumps(FORUM_ITEM) + "\n" + json.dumps(COMMENT_ITEM) + "\n",
        encoding="utf-8",
    )
    forum, comments = rt.load_items(jsonl)
    assert forum is not None
    assert forum["reg_title"] == FORUM_ITEM["reg_title"]
    assert len(comments) == 1
    assert comments[0]["comment_id"] == "87914"


def test_load_items_no_forum(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(json.dumps(COMMENT_ITEM) + "\n", encoding="utf-8")
    forum, comments = rt.load_items(jsonl)
    assert forum is None
    assert len(comments) == 1


def test_load_items_skips_blank_lines(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text("\n" + json.dumps(COMMENT_ITEM) + "\n\n", encoding="utf-8")
    _, comments = rt.load_items(jsonl)
    assert len(comments) == 1


# ---------------------------------------------------------------------------
# build_messages

def test_truncation_applied():
    long_comment = {**COMMENT_ITEM, "text": "x" * 7000}
    messages, truncated = rt.build_messages(long_comment, FORUM_ITEM)
    assert truncated is True
    assert "... [truncated]" in messages[1]["content"]
    assert messages[1]["content"].count("x") == rt.MAX_COMMENT_CHARS


def test_no_truncation_for_short_comment():
    _, truncated = rt.build_messages(COMMENT_ITEM, FORUM_ITEM)
    assert truncated is False


def test_empty_text_fallback():
    empty = {**COMMENT_ITEM, "text": ""}
    messages, truncated = rt.build_messages(empty, FORUM_ITEM)
    assert "[No body text provided]" in messages[1]["content"]
    assert truncated is False


def test_none_text_fallback():
    none_text = {**COMMENT_ITEM, "text": None}
    messages, _ = rt.build_messages(none_text, FORUM_ITEM)
    assert "[No body text provided]" in messages[1]["content"]


def test_missing_forum_uses_unknown_context():
    messages, _ = rt.build_messages(COMMENT_ITEM, None)
    assert "[unknown]" in messages[1]["content"]


def test_reg_context_included_in_prompt():
    messages, _ = rt.build_messages(COMMENT_ITEM, FORUM_ITEM)
    assert FORUM_ITEM["reg_title"] in messages[1]["content"]
    assert "HB 145" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Output record schema

def test_output_record_all_keys_present():
    record = rt.analyze_comment(_mock_client(), COMMENT_ITEM, FORUM_ITEM, "run-123", "gpt-4o")
    required = {
        "run_id", "forum_id", "comment_id", "analyzed_at", "model", "prompt_version",
        "stance", "stance_confidence", "stance_rationale", "tone", "tags",
        "input_title", "truncated", "error",
    }
    assert required == set(record.keys())


def test_output_record_correct_types():
    record = rt.analyze_comment(_mock_client(), COMMENT_ITEM, FORUM_ITEM, "run-123", "gpt-4o")
    assert record["stance"] == "support"
    assert isinstance(record["stance_confidence"], float)
    assert isinstance(record["tags"], list)
    assert record["truncated"] is False
    assert record["error"] is None


def test_output_record_metadata():
    record = rt.analyze_comment(_mock_client(), COMMENT_ITEM, FORUM_ITEM, "run-123", "gpt-4o")
    assert record["run_id"] == "run-123"
    assert record["forum_id"] == "452"
    assert record["comment_id"] == "87914"
    assert record["model"] == "gpt-4o"
    assert record["prompt_version"] == rt.PROMPT_VERSION
    assert record["input_title"] == COMMENT_ITEM["title"]


# ---------------------------------------------------------------------------
# Error handling

def test_error_record_on_api_failure():
    import openai as _openai
    client = MagicMock()
    client.chat.completions.create.side_effect = _openai.RateLimitError(
        "rate limit", response=MagicMock(status_code=429), body={}
    )
    record = rt.analyze_comment(client, COMMENT_ITEM, FORUM_ITEM, "run-123", "gpt-4o")
    assert record["error"] is not None
    assert record["stance"] is None
    assert record["tone"] is None
    assert record["tags"] is None


def test_error_record_on_bad_json():
    record = rt.analyze_comment(_mock_client("not valid json{{{"), COMMENT_ITEM, FORUM_ITEM, "run-123", "gpt-4o")
    assert record["error"] is not None
    assert record["stance"] is None


# ---------------------------------------------------------------------------
# run_id consistency

def test_run_id_is_shared_across_records():
    client = _mock_client()
    run_id = "fixed-run-id"
    r1 = rt.analyze_comment(client, COMMENT_ITEM, FORUM_ITEM, run_id, "gpt-4o")
    r2 = rt.analyze_comment(client, {**COMMENT_ITEM, "comment_id": "99999"}, FORUM_ITEM, run_id, "gpt-4o")
    assert r1["run_id"] == r2["run_id"] == run_id


# ---------------------------------------------------------------------------
# Filename helpers

def test_scrape_ts_extracted_from_filename():
    p = Path("output/forum452_comments_2026-05-05T17-33-54+00-00.jsonl")
    assert rt._scrape_ts_from_filename(p) == "2026-05-05T17-33-54+00-00"


def test_scrape_ts_fallback_for_unknown_filename():
    assert rt._scrape_ts_from_filename(Path("output/somefile.jsonl")) == "unknown"
