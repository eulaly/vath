"""Unit tests for analysis/gpt4o/analysis_batch.py — no real API calls."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "gpt4o"))
import analysis_batch as bt


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

RAW_SUCCESS_LINE = {
    "id": "batch_req_001",
    "custom_id": "comment_87914",
    "response": {
        "status_code": 200,
        "request_id": "req_abc",
        "body": {
            "id": "chatcmpl-xyz",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "stance": "support",
                        "stance_confidence": 0.95,
                        "stance_rationale": "Commenter explicitly endorses the policy.",
                        "tone": "positive",
                        "tags": ["student safety"],
                    }),
                },
                "finish_reason": "stop",
            }],
        },
    },
    "error": None,
}

RAW_ERROR_LINE = {
    "id": "batch_req_002",
    "custom_id": "comment_87914",
    "response": None,
    "error": {"code": "batch_expired", "message": "This request could not be executed."},
}

RAW_HTTP_ERROR_LINE = {
    "id": "batch_req_003",
    "custom_id": "comment_87914",
    "response": {"status_code": 400, "body": {}},
    "error": None,
}

COMMENT_LOOKUP = {"87914": COMMENT_ITEM}
ANALYZED_AT = "2026-05-05T18:00:00+00:00"
RUN_ID = "test-run-id-123"
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Prompt versioning (batch reads the same prompt file)

def test_prompt_version_is_7_hex_chars():
    assert len(bt.PROMPT_VERSION) == 7
    assert all(c in "0123456789abcdef" for c in bt.PROMPT_VERSION)


def test_prompt_version_matches_realtime():
    """Both scripts must derive the same PROMPT_VERSION from the same file."""
    import analysis_realtime as rt
    assert bt.PROMPT_VERSION == rt.PROMPT_VERSION


# ---------------------------------------------------------------------------
# custom_id helpers

def test_custom_id_from():
    assert bt.custom_id_from("87914") == "comment_87914"


def test_parse_custom_id():
    assert bt.parse_custom_id("comment_87914") == "87914"


def test_custom_id_round_trip():
    cid = "12345"
    assert bt.parse_custom_id(bt.custom_id_from(cid)) == cid


# ---------------------------------------------------------------------------
# build_batch_request_line

def test_batch_request_line_structure():
    line = bt.build_batch_request_line(COMMENT_ITEM, FORUM_ITEM, "gpt-4o")
    assert line["custom_id"] == "comment_87914"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"] == "gpt-4o"
    assert line["body"]["temperature"] == 0.0
    assert line["body"]["response_format"] == {"type": "json_object"}
    messages = line["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_batch_request_line_includes_reg_context():
    line = bt.build_batch_request_line(COMMENT_ITEM, FORUM_ITEM, "gpt-4o")
    user_content = line["body"]["messages"][1]["content"]
    assert "Model Policies for Transgender Students" in user_content
    assert "HB 145" in user_content


def test_batch_request_line_truncation():
    long_comment = {**COMMENT_ITEM, "text": "x" * 7000}
    line = bt.build_batch_request_line(long_comment, FORUM_ITEM, "gpt-4o")
    user_content = line["body"]["messages"][1]["content"]
    assert "... [truncated]" in user_content
    assert user_content.count("x") == bt.MAX_COMMENT_CHARS


# ---------------------------------------------------------------------------
# normalize_output_line — success

def test_normalize_success_all_keys():
    record = bt.normalize_output_line(RAW_SUCCESS_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    required = {
        "run_id", "forum_id", "comment_id", "analyzed_at", "model", "prompt_version",
        "stance", "stance_confidence", "stance_rationale", "tone", "tags",
        "input_title", "truncated", "error",
    }
    assert required == set(record.keys())


def test_normalize_success_values():
    record = bt.normalize_output_line(RAW_SUCCESS_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["stance"] == "support"
    assert record["tone"] == "positive"
    assert record["comment_id"] == "87914"
    assert record["run_id"] == RUN_ID
    assert record["analyzed_at"] == ANALYZED_AT
    assert record["error"] is None
    assert record["truncated"] is False


def test_normalize_success_input_title():
    record = bt.normalize_output_line(RAW_SUCCESS_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["input_title"] == COMMENT_ITEM["title"]


# ---------------------------------------------------------------------------
# normalize_output_line — errors

def test_normalize_batch_expired_error():
    record = bt.normalize_output_line(RAW_ERROR_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["error"] is not None
    assert "could not be executed" in record["error"]
    assert record["stance"] is None
    assert record["tone"] is None


def test_normalize_http_error():
    record = bt.normalize_output_line(RAW_HTTP_ERROR_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["error"] is not None
    assert record["stance"] is None


def test_normalize_malformed_json_in_response():
    bad_line = {
        "id": "batch_req_004",
        "custom_id": "comment_87914",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": "not valid json{{{"}}]},
        },
        "error": None,
    }
    record = bt.normalize_output_line(bad_line, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["error"] is not None
    assert record["stance"] is None


def test_normalize_unknown_comment_id():
    """A custom_id not in lookup yields empty forum_id and title but doesn't crash."""
    record = bt.normalize_output_line(RAW_SUCCESS_LINE, {}, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert record["comment_id"] == "87914"
    assert record["forum_id"] == ""
    assert record["input_title"] == ""


# ---------------------------------------------------------------------------
# Manifest

def test_make_manifest_all_keys():
    m = bt.make_manifest(
        run_id=RUN_ID,
        input_filename="output/forum452.jsonl",
        input_sha256="abc123",
        model="gpt-4o",
        batch_id="batch_xyz",
        records_submitted=100,
        request_filename="analysis/gpt4o/requests/test-run-id-123.jsonl",
    )
    required = {
        "run_id", "input_filename", "input_sha256", "prompt_hash", "model",
        "batch_id", "records_submitted", "records_completed", "records_failed",
        "request_filename", "raw_output_filename", "normalized_output_filename",
        "created_at", "completed_at",
    }
    assert required == set(m.keys())


def test_make_manifest_initial_nulls():
    m = bt.make_manifest(
        run_id=RUN_ID, input_filename="f", input_sha256="s",
        model="gpt-4o", batch_id="b", records_submitted=10, request_filename="r",
    )
    assert m["records_completed"] is None
    assert m["records_failed"] is None
    assert m["raw_output_filename"] is None
    assert m["normalized_output_filename"] is None
    assert m["completed_at"] is None
    assert m["prompt_hash"] == bt.PROMPT_VERSION


def test_manifest_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "RUNS_DIR", tmp_path)
    m = bt.make_manifest(
        run_id=RUN_ID, input_filename="f", input_sha256="s",
        model="gpt-4o", batch_id="b", records_submitted=42, request_filename="r",
    )
    bt.save_manifest(m)
    loaded = bt.load_manifest(RUN_ID)
    assert loaded == m


# ---------------------------------------------------------------------------
# estimate_tokens

def test_estimate_tokens_returns_positive_int():
    messages = [{"role": "system", "content": "hello"}, {"role": "user", "content": "world"}]
    result = bt.estimate_tokens(messages, "gpt-4o-mini")
    assert isinstance(result, int)
    assert result > 0


def test_estimate_tokens_longer_content_is_larger():
    short_msg = [{"role": "user", "content": "hi"}]
    long_msg  = [{"role": "user", "content": "hi " * 500}]
    assert bt.estimate_tokens(long_msg, "gpt-4o-mini") > bt.estimate_tokens(short_msg, "gpt-4o-mini")


def test_estimate_tokens_fallback_without_tiktoken(monkeypatch):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "tiktoken", None)
    messages = [{"role": "user", "content": "x" * 300}]
    result = bt.estimate_tokens(messages, "gpt-4o")
    assert result == 4 + 300 // 3


# ---------------------------------------------------------------------------
# chunk_comments_by_tokens

def test_chunk_single_chunk_for_small_input(monkeypatch):
    monkeypatch.setattr(bt, "MODEL_LIMITS", {"gpt-4o-mini": 10_000_000})
    comments = [COMMENT_ITEM, {**COMMENT_ITEM, "comment_id": "99999"}]
    chunks = bt.chunk_comments_by_tokens(comments, FORUM_ITEM, "gpt-4o-mini")
    assert len(chunks) == 1
    assert len(chunks[0]) == 2


def test_chunk_splits_when_over_limit(monkeypatch):
    monkeypatch.setattr(bt, "MODEL_LIMITS", {"gpt-4o-mini": 1})
    comments = [
        COMMENT_ITEM,
        {**COMMENT_ITEM, "comment_id": "99999"},
        {**COMMENT_ITEM, "comment_id": "88888"},
    ]
    chunks = bt.chunk_comments_by_tokens(comments, FORUM_ITEM, "gpt-4o-mini")
    assert len(chunks) == len(comments)


def test_chunk_preserves_all_comments(monkeypatch):
    monkeypatch.setattr(bt, "MODEL_LIMITS", {"gpt-4o-mini": 200})
    comments = [{**COMMENT_ITEM, "comment_id": str(i)} for i in range(10)]
    chunks = bt.chunk_comments_by_tokens(comments, FORUM_ITEM, "gpt-4o-mini")
    flat = [c for chunk in chunks for c in chunk]
    assert len(flat) == 10


def test_model_limits_has_required_models():
    for model in ("gpt-4o", "gpt-4o-mini", "gpt-5.4", "gpt-5.4-mini", "gpt-o4-mini"):
        assert model in bt.MODEL_LIMITS, f"{model} missing from MODEL_LIMITS"
