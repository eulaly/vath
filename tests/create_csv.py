"""Unit tests for analysis/create_csv.py — no external API calls."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
import create_csv as cc


# ---------------------------------------------------------------------------
# Helpers

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


RAW_ROWS = [
    {"forum_id": "452", "comment_id": "1", "title": "Support", "text": "I support.", "date": "2021-01-01", "author": "Alice"},
    {"forum_id": "452", "comment_id": "2", "title": "Oppose",  "text": "I oppose.",  "date": "2021-01-02", "author": "Bob"},
    {"forum_id": "452", "comment_id": "3", "title": "Neutral", "text": "No opinion.","date": "2021-01-03", "author": "Carol"},
]

ANALYSIS_ROWS = [
    {"comment_id": "1", "stance": "support", "stance_confidence": 0.9, "stance_rationale": "clear support",
     "tone": "neutral", "tags": '["policy"]', "error": None, "truncated": False,
     "analyzed_at": "2021-01-10", "prompt_version": "1", "model": "gpt-4o-mini"},
    {"comment_id": "2", "stance": "oppose",  "stance_confidence": 0.8, "stance_rationale": "clear oppose",
     "tone": "negative", "tags": '[]', "error": None, "truncated": False,
     "analyzed_at": "2021-01-10", "prompt_version": "1", "model": "gpt-4o-mini"},
]


# ---------------------------------------------------------------------------
# load_raw

def test_load_raw_returns_raw_cols(tmp_path):
    p = tmp_path / "forum.jsonl"
    _write_jsonl(p, RAW_ROWS)
    df = cc.load_raw(p)
    assert list(df.columns) == cc.RAW_COLS


def test_load_raw_row_count(tmp_path):
    p = tmp_path / "forum.jsonl"
    _write_jsonl(p, RAW_ROWS)
    df = cc.load_raw(p)
    assert len(df) == 3


def test_load_raw_skips_non_comment_rows(tmp_path):
    """Rows without comment_id (e.g. forum metadata) are dropped."""
    rows = RAW_ROWS + [{"forum_id": "452", "reg_title": "Metadata row"}]
    p = tmp_path / "forum.jsonl"
    _write_jsonl(p, rows)
    df = cc.load_raw(p)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# load_analysis

def test_load_analysis_returns_analysis_cols(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _write_jsonl(jobs / "job1-output.jsonl", ANALYSIS_ROWS)
    df = cc.load_analysis(jobs)
    expected = ["comment_id"] + cc.ANALYSIS_COLS
    assert list(df.columns) == expected


def test_load_analysis_skips_raw_files(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _write_jsonl(jobs / "job1-output.jsonl", ANALYSIS_ROWS)
    _write_jsonl(jobs / "job1-output-raw.jsonl", ANALYSIS_ROWS)  # should be ignored
    df = cc.load_analysis(jobs)
    assert len(df) == len(ANALYSIS_ROWS)


def test_load_analysis_concatenates_multiple_files(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _write_jsonl(jobs / "job1-output.jsonl", [ANALYSIS_ROWS[0]])
    _write_jsonl(jobs / "job2-output.jsonl", [ANALYSIS_ROWS[1]])
    df = cc.load_analysis(jobs)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# join

def test_join_all_raw_preserved(tmp_path):
    """Left join: all raw comments appear in output, even without analysis."""
    raw = pd.DataFrame(RAW_ROWS)[cc.RAW_COLS]
    analysis = pd.DataFrame(ANALYSIS_ROWS)
    for col in cc.ANALYSIS_COLS:
        if col not in analysis.columns:
            analysis[col] = None
    analysis = analysis[["comment_id"] + cc.ANALYSIS_COLS]

    merged = cc.join(raw, analysis)
    assert len(merged) == 3  # all 3 raw rows, even comment_id=3 with no analysis


def test_join_unanalyzed_row_has_null_stance(tmp_path):
    raw = pd.DataFrame(RAW_ROWS)[cc.RAW_COLS]
    analysis = pd.DataFrame(ANALYSIS_ROWS)
    for col in cc.ANALYSIS_COLS:
        if col not in analysis.columns:
            analysis[col] = None
    analysis = analysis[["comment_id"] + cc.ANALYSIS_COLS]

    merged = cc.join(raw, analysis)
    unanalyzed = merged[merged["comment_id"] == "3"]
    assert pd.isna(unanalyzed.iloc[0]["stance"])


def test_join_column_order(tmp_path):
    raw = pd.DataFrame(RAW_ROWS)[cc.RAW_COLS]
    analysis = pd.DataFrame(ANALYSIS_ROWS)
    for col in cc.ANALYSIS_COLS:
        if col not in analysis.columns:
            analysis[col] = None
    analysis = analysis[["comment_id"] + cc.ANALYSIS_COLS]

    merged = cc.join(raw, analysis)
    assert list(merged.columns) == cc.OUTPUT_COLS


# ---------------------------------------------------------------------------
# End-to-end: write + read CSV

def test_csv_written_correctly(tmp_path):
    raw_path = tmp_path / "forum.jsonl"
    _write_jsonl(raw_path, RAW_ROWS)

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _write_jsonl(jobs / "job1-output.jsonl", ANALYSIS_ROWS)

    out = tmp_path / "review.csv"
    raw      = cc.load_raw(raw_path)
    analysis = cc.load_analysis(jobs)
    merged   = cc.join(raw, analysis)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    loaded = pd.read_csv(out)
    assert len(loaded) == 3
    assert list(loaded.columns) == cc.OUTPUT_COLS
