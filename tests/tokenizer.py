"""Unit tests for analysis/gpt4o/tokenizer.py — no real API calls."""

import io
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "gpt4o"))
import tokenizer as tk
import analysis_batch as ab


# ---------------------------------------------------------------------------
# Fixtures

FORUM_ITEM = {
    "forum_id": "452",
    "reg_title": "Model Policies for Transgender Students",
    "reg_desc": "Guidance developed in response to HB 145.",
}

COMMENT_A = {
    "forum_id": "452",
    "comment_id": "100",
    "author": "Alice",
    "date": "2021-01-04T09:15:00",
    "title": "Support",
    "text": "I support this policy.",
}

COMMENT_B = {
    "forum_id": "452",
    "comment_id": "101",
    "author": "Bob",
    "date": "2021-01-05T10:00:00",
    "title": "Oppose",
    "text": "I oppose this policy.",
}

COMMENTS = [COMMENT_A, COMMENT_B]
PROMPT_HASH = "abc1234"
INPUT_FILE = "output/f452.jsonl"
INPUT_SHA256 = "deadbeef" * 8
PROMPT_FILE = "analysis/prompt-1.txt"


def _make_report(total_tokens=10_000):
    return tk.compute_report(
        COMMENTS, FORUM_ITEM, PROMPT_HASH, INPUT_FILE, INPUT_SHA256, PROMPT_FILE
    )


# ---------------------------------------------------------------------------
# compute_report: required top-level keys

def test_report_has_top_level_keys():
    report = _make_report()
    required = {"prompt", "prompt_hash", "input_file", "input_sha256",
                "total_comments", "input_tokens"}
    assert required.issubset(set(report.keys()))


def test_report_metadata_values():
    report = _make_report()
    assert report["prompt"] == PROMPT_FILE
    assert report["prompt_hash"] == PROMPT_HASH
    assert report["input_file"] == INPUT_FILE
    assert report["input_sha256"] == INPUT_SHA256
    assert report["total_comments"] == 2


def test_report_input_tokens_positive():
    report = _make_report()
    assert isinstance(report["input_tokens"], int)
    assert report["input_tokens"] > 0


# ---------------------------------------------------------------------------
# compute_report: per-model entries

def test_report_has_per_model_keys():
    report = _make_report()
    for model in ab.MODEL_LIMITS:
        assert model in report, f"Model {model} missing from report"
        assert isinstance(report[model], dict)


def test_report_per_model_has_required_fields():
    report = _make_report()
    for model in ab.MODEL_LIMITS:
        m = report[model]
        assert "jobs" in m
        assert "cost_$" in m
        assert "est_queue_days" in m


def test_report_jobs_at_least_one():
    report = _make_report()
    for model in ab.MODEL_LIMITS:
        assert report[model]["jobs"] >= 1


# ---------------------------------------------------------------------------
# compute_report: calculation accuracy

def test_cost_calculation():
    """cost_$ = total_tokens / 1M * pricing_rate"""
    report = _make_report()
    total = report["input_tokens"]
    for model in ab.MODEL_LIMITS:
        expected_cost = round(total / 1_000_000 * tk.MODEL_PRICING.get(model, 0.0), 4)
        assert report[model]["cost_$"] == pytest.approx(expected_cost, abs=1e-6)


def test_est_queue_days_calculation():
    """est_queue_days = total_tokens / tpd (rounded to 2 decimal places)"""
    report = _make_report()
    total = report["input_tokens"]
    for model, tpd in ab.MODEL_LIMITS.items():
        expected = round(total / tpd, 2)
        assert report[model]["est_queue_days"] == pytest.approx(expected, abs=1e-4)


def test_jobs_ceiling_division():
    """jobs = ceil(total_tokens / (tpd * _LIMIT_BUFFER))"""
    report = _make_report()
    total = report["input_tokens"]
    for model, tpd in ab.MODEL_LIMITS.items():
        effective = int(tpd * ab._LIMIT_BUFFER)
        expected = math.ceil(total / effective)
        assert report[model]["jobs"] == expected


def test_more_comments_increases_tokens():
    """More comments → more input_tokens."""
    few = tk.compute_report([COMMENT_A], FORUM_ITEM, PROMPT_HASH, INPUT_FILE, INPUT_SHA256, PROMPT_FILE)
    many = tk.compute_report(COMMENTS, FORUM_ITEM, PROMPT_HASH, INPUT_FILE, INPUT_SHA256, PROMPT_FILE)
    assert many["input_tokens"] > few["input_tokens"]


# ---------------------------------------------------------------------------
# MODEL_PRICING coverage

def test_model_pricing_has_required_models():
    for model in ("gpt-4o", "gpt-4o-mini", "gpt-5.4", "gpt-5.4-mini", "gpt-o4-mini"):
        assert model in tk.MODEL_PRICING, f"{model} missing from MODEL_PRICING"


def test_model_pricing_values_positive():
    for model, price in tk.MODEL_PRICING.items():
        assert price > 0, f"{model} has non-positive price"


# ---------------------------------------------------------------------------
# print_table: runs without error, produces output

def test_print_table_runs():
    report = _make_report()
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        tk.print_table(report)
    output = buf.getvalue()
    assert "gpt-4o" in output
    assert "gpt-4o-mini" in output


def test_print_table_shows_all_models():
    report = _make_report()
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        tk.print_table(report)
    output = buf.getvalue()
    for model in ab.MODEL_LIMITS:
        assert model in output, f"{model} not shown in print_table output"


def test_print_table_highlights_recommended():
    """When a single-job cheapest model exists, table marks it as recommended."""
    report = _make_report()
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        tk.print_table(report)
    output = buf.getvalue()
    assert "recommended" in output


# ---------------------------------------------------------------------------
# report.json round-trip (write → read)

def test_report_json_roundtrip(tmp_path):
    report = _make_report()
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["total_comments"] == report["total_comments"]
    assert loaded["input_tokens"] == report["input_tokens"]
    assert loaded["gpt-4o-mini"]["jobs"] == report["gpt-4o-mini"]["jobs"]
