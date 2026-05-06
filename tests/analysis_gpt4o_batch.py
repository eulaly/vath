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

# Minimal status.json for testing job logic
def _make_status(jobs_override=None):
    jobs = jobs_override or [
        {"job_num": 1, "run_id": "r1", "status": "pending", "batch_id": None,
         "records_submitted": 60, "records_completed": None, "records_failed": None,
         "submitted_at": None, "completed_at": None},
    ]
    return {
        "model": "gpt-4o-mini", "prompt_hash": "abc1234",
        "input_file": "output/f452.jsonl", "input_sha256": "sha",
        "total_comments": 100, "input_tokens": 50_000,
        "est_queue_days": 0.025, "cost_$": 0.01,
        "total_jobs": len(jobs), "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Prompt versioning

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


# ---------------------------------------------------------------------------
# status.json helpers

def test_status_save_load_roundtrip(tmp_path):
    status = _make_status()
    bt.save_status(status, tmp_path)
    loaded = bt.load_status(tmp_path)
    assert loaded == status


# ---------------------------------------------------------------------------
# _find_next_eligible_job

def test_find_next_eligible_job_first_job_pending():
    jobs = _make_status()["jobs"]
    target, warning = bt._find_next_eligible_job(jobs)
    assert target["job_num"] == 1
    assert warning is None


def test_find_next_eligible_job_after_completed():
    jobs = [
        {"job_num": 1, "status": "completed", "batch_id": "b1",
         "records_submitted": 60, "records_completed": 60, "records_failed": 0,
         "submitted_at": "t", "completed_at": "t", "run_id": "r1"},
        {"job_num": 2, "status": "pending", "batch_id": None,
         "records_submitted": 40, "records_completed": None, "records_failed": None,
         "submitted_at": None, "completed_at": None, "run_id": "r2"},
    ]
    target, warning = bt._find_next_eligible_job(jobs)
    assert target["job_num"] == 2
    assert warning is None


def test_find_next_eligible_job_blocked_by_in_progress():
    jobs = [
        {"job_num": 1, "status": "in_progress", "batch_id": "b1",
         "records_submitted": 60, "records_completed": None, "records_failed": None,
         "submitted_at": "t", "completed_at": None, "run_id": "r1"},
        {"job_num": 2, "status": "pending", "batch_id": None,
         "records_submitted": 40, "records_completed": None, "records_failed": None,
         "submitted_at": None, "completed_at": None, "run_id": "r2"},
    ]
    target, warning = bt._find_next_eligible_job(jobs)
    assert target is None
    assert warning is not None
    assert "in_progress" in warning


def test_find_next_eligible_job_all_completed():
    jobs = [
        {"job_num": 1, "status": "completed", "batch_id": "b1",
         "records_submitted": 60, "records_completed": 60, "records_failed": 0,
         "submitted_at": "t", "completed_at": "t", "run_id": "r1"},
    ]
    target, warning = bt._find_next_eligible_job(jobs)
    assert target is None
    assert warning is None


def test_resume_from_status_json(tmp_path):
    """Reload a status.json with one completed job and find the next pending job."""
    jobs = [
        {"job_num": 1, "run_id": "r1", "status": "completed", "batch_id": "b1",
         "records_submitted": 60, "records_completed": 58, "records_failed": 2,
         "submitted_at": "2026-05-06T10:00:00+00:00", "completed_at": "2026-05-06T11:00:00+00:00"},
        {"job_num": 2, "run_id": "r2", "status": "pending", "batch_id": None,
         "records_submitted": 40, "records_completed": None, "records_failed": None,
         "submitted_at": None, "completed_at": None},
    ]
    bt.save_status(_make_status(jobs), tmp_path)
    loaded = bt.load_status(tmp_path)
    target, warning = bt._find_next_eligible_job(loaded["jobs"])
    assert target["job_num"] == 2
    assert warning is None


# ---------------------------------------------------------------------------
# normalize: out-of-order and duplicate custom_id

def test_out_of_order_output_reconciled_by_custom_id():
    """Raw lines processed in any order are mapped to the correct comment."""
    c2 = {**COMMENT_ITEM, "comment_id": "99999", "title": "Second comment"}
    lookup = {COMMENT_ITEM["comment_id"]: COMMENT_ITEM, "99999": c2}

    line_for_99999 = {
        **RAW_SUCCESS_LINE,
        "custom_id": "comment_99999",
    }
    line_for_87914 = RAW_SUCCESS_LINE

    r1 = bt.normalize_output_line(line_for_99999, lookup, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    r2 = bt.normalize_output_line(line_for_87914, lookup, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)

    assert r1["comment_id"] == "99999"
    assert r1["input_title"] == "Second comment"
    assert r2["comment_id"] == "87914"
    assert r2["input_title"] == COMMENT_ITEM["title"]


def test_duplicate_custom_id_both_produce_valid_records():
    """Two raw lines with the same custom_id each produce a valid record."""
    r1 = bt.normalize_output_line(RAW_SUCCESS_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    r2 = bt.normalize_output_line(RAW_SUCCESS_LINE, COMMENT_LOOKUP, RUN_ID, ANALYZED_AT, MODEL, bt.PROMPT_VERSION)
    assert r1["comment_id"] == r2["comment_id"] == "87914"
    assert r1["error"] is None
    assert r2["error"] is None
