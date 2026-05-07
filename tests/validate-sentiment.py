"""Unit tests for analysis/validate.py — no file I/O beyond tmp_path."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

try:
    import pandas as pd
except ImportError:
    pytest.skip("pandas not installed", allow_module_level=True)

import validate as vl

# ---------------------------------------------------------------------------
# Fixtures


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


RAW_ROWS = [
    {"forum_id": "452", "comment_id": "1", "title": "Support it",
     "text": "I support this.", "date": "2021-01-04T09:00:00", "author": "Alice"},
    {"forum_id": "452", "comment_id": "2", "title": "Oppose it",
     "text": "I oppose this.", "date": "2021-01-05T10:00:00", "author": "Bob"},
    {"forum_id": "452", "comment_id": "3", "title": "Neutral",
     "text": "No opinion.", "date": "2021-01-06T11:00:00", "author": "Carol"},
]

ANALYSIS_ROWS = [
    {"run_id": "r1", "forum_id": "452", "comment_id": "1", "input_title": "Support it",
     "analyzed_at": "2026-05-06T12:00:00+00:00", "model": "gpt-5.4-mini",
     "prompt_version": "abc1234", "stance": "support", "stance_confidence": 0.95,
     "stance_rationale": "Commenter says 'I support'.", "tone": "positive",
     "tags": ["student safety"], "truncated": False, "error": None},
    {"run_id": "r1", "forum_id": "452", "comment_id": "2", "input_title": "Oppose it",
     "analyzed_at": "2026-05-06T12:00:00+00:00", "model": "gpt-5.4-mini",
     "prompt_version": "abc1234", "stance": "oppose", "stance_confidence": 0.90,
     "stance_rationale": "Commenter says 'I oppose'.", "tone": "negative",
     "tags": [], "truncated": False, "error": None},
]

FORUM_ROW = {"forum_id": "452", "reg_title": "Policy X", "reg_desc": "Guidance on Y."}


@pytest.fixture()
def raw_jsonl(tmp_path) -> Path:
    p = tmp_path / "f452.jsonl"
    _write_jsonl(p, [FORUM_ROW] + RAW_ROWS)
    return p


@pytest.fixture()
def jobs_dir(tmp_path) -> Path:
    d = tmp_path / "jobs" / "f452-1"
    d.mkdir(parents=True)
    _write_jsonl(d / "job1-output.jsonl", ANALYSIS_ROWS)
    return d


# ---------------------------------------------------------------------------
# load_raw


def test_load_raw_returns_only_comments(raw_jsonl):
    df = vl.load_raw(raw_jsonl)
    assert len(df) == 3
    assert set(df.columns) == set(vl.RAW_COLS)


def test_load_raw_correct_columns(raw_jsonl):
    df = vl.load_raw(raw_jsonl)
    for col in vl.RAW_COLS:
        assert col in df.columns


def test_load_raw_skips_forum_item(raw_jsonl):
    df = vl.load_raw(raw_jsonl)
    assert "reg_title" not in df.columns


# ---------------------------------------------------------------------------
# load_analysis


def test_load_analysis_skips_raw_files(tmp_path):
    d = tmp_path / "jobs" / "f452-1"
    d.mkdir(parents=True)
    _write_jsonl(d / "job1-output-raw.jsonl", ANALYSIS_ROWS)   # should be ignored
    _write_jsonl(d / "job1-output.jsonl", ANALYSIS_ROWS)
    df = vl.load_analysis(d)
    assert len(df) == len(ANALYSIS_ROWS)


def test_load_analysis_concatenates_multiple_files(tmp_path):
    d = tmp_path / "jobs" / "f452-1"
    d.mkdir(parents=True)
    _write_jsonl(d / "job1-output.jsonl", [ANALYSIS_ROWS[0]])
    _write_jsonl(d / "job2-output.jsonl", [ANALYSIS_ROWS[1]])
    df = vl.load_analysis(d)
    assert len(df) == 2


def test_load_analysis_tags_serialized_as_json(jobs_dir):
    df = vl.load_analysis(jobs_dir)
    tags_val = df.loc[df["comment_id"] == "1", "tags"].iloc[0]
    assert isinstance(tags_val, str)
    assert json.loads(tags_val) == ["student safety"]


def test_load_analysis_empty_tags_serialized(jobs_dir):
    df = vl.load_analysis(jobs_dir)
    tags_val = df.loc[df["comment_id"] == "2", "tags"].iloc[0]
    assert json.loads(tags_val) == []


# ---------------------------------------------------------------------------
# join — by comment_id, not index


def test_join_by_comment_id_not_index(raw_jsonl, jobs_dir):
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    # Shuffle raw order so comment_id ordering differs from index
    raw = raw.sample(frac=1, random_state=42).reset_index(drop=True)
    merged = vl.join(raw, analysis)
    row_1 = merged[merged["comment_id"] == "1"].iloc[0]
    assert row_1["stance"] == "support"
    assert row_1["author"] == "Alice"


def test_join_unanalyzed_comment_has_null_stance(raw_jsonl, jobs_dir):
    """Comment 3 is in raw but not in analysis — stance should be NaN."""
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    row_3 = merged[merged["comment_id"] == "3"].iloc[0]
    assert pd.isna(row_3["stance"])


def test_join_preserves_all_raw_comments(raw_jsonl, jobs_dir):
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    assert len(merged) == len(raw)


def test_join_output_columns_in_order(raw_jsonl, jobs_dir):
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    assert list(merged.columns) == vl.OUTPUT_COLS


# ---------------------------------------------------------------------------
# Duplicate comment_id handling


def test_duplicate_raw_id_flagged(raw_jsonl, jobs_dir):
    raw      = vl.load_raw(raw_jsonl)
    # Manually duplicate a row
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    # join still produces a row for each raw row (left join)
    assert len(merged) == len(raw)
    assert raw["comment_id"].duplicated().sum() == 1


def test_duplicate_analysis_id_produces_extra_rows(raw_jsonl, tmp_path):
    """Two analysis records for the same comment_id create two joined rows."""
    d = tmp_path / "jobs" / "f452-dup"
    d.mkdir(parents=True)
    dup_rows = [ANALYSIS_ROWS[0], {**ANALYSIS_ROWS[0], "stance": "oppose"}]
    _write_jsonl(d / "job1-output.jsonl", dup_rows)
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(d)
    merged   = vl.join(raw, analysis)
    assert len(merged[merged["comment_id"] == "1"]) == 2


# ---------------------------------------------------------------------------
# Validation counts (smoke test — just confirm it runs without error)


def test_print_validation_runs(raw_jsonl, jobs_dir, capsys):
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    vl.print_validation(raw, analysis, merged)
    out = capsys.readouterr().out
    assert "Raw comments" in out
    assert "Stance counts" in out
    assert "Tone counts" in out


# ---------------------------------------------------------------------------
# CSV output


def test_csv_written_to_jobs_dir(raw_jsonl, jobs_dir, tmp_path):
    raw      = vl.load_raw(raw_jsonl)
    analysis = vl.load_analysis(jobs_dir)
    merged   = vl.join(raw, analysis)
    out_path = jobs_dir / "review.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    assert out_path.exists()
    loaded = pd.read_csv(out_path, encoding="utf-8-sig")
    assert list(loaded.columns) == vl.OUTPUT_COLS
    assert len(loaded) == len(raw)
