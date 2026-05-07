#!/usr/bin/env python3
"""analysis/create_csv.py — join raw scrape with analysis output for review."""

import argparse
from pathlib import Path

import pandas as pd

RAW_COLS = ["forum_id", "comment_id", "title", "text", "date", "author"]
ANALYSIS_COLS = [
    "stance", "stance_confidence", "stance_rationale", "tone", "tags",
    "error", "truncated", "analyzed_at", "prompt_version", "model",
]
OUTPUT_COLS = RAW_COLS + ANALYSIS_COLS


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df[df["comment_id"].notna()] # rm first item (forum, not comment)
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = None
    return df[RAW_COLS].copy()


def load_analysis(jobs_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in jobs_dir.glob("job*-output.jsonl") if "-raw" not in p.name)
    df = pd.concat([pd.read_json(p, lines=True) for p in files], ignore_index=True)
    for col in ANALYSIS_COLS:
        if col not in df.columns:
            df[col] = None
    return df[["comment_id"] + ANALYSIS_COLS].copy()


def join(raw: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    return raw.merge(analysis, on="comment_id", how="left")[OUTPUT_COLS]


def print_counts(raw: pd.DataFrame, analysis: pd.DataFrame, merged: pd.DataFrame) -> None:
    print(f"\nRaw comments  : {len(raw):,}")
    print(f"Analyzed      : {len(analysis):,}")
    print(f"Joined        : {merged['stance'].notna().sum():,}")
    print(f"Unanalyzed    : {merged['stance'].isna().sum():,}")
    print(f"Errors        : {analysis['error'].notna().sum():,}")
    print(f"Dup IDs (raw) : {raw['comment_id'].duplicated().sum():,}")
    print(f"\nStance:\n{analysis['stance'].value_counts(dropna=False).to_string()}")
    print(f"\nTone:\n{analysis['tone'].value_counts(dropna=False).to_string()}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Join raw scrape JSONL with analysis output; write review CSV."
    )
    p.add_argument("input",    help="Raw scrape JSONL (e.g. output/f452.jsonl)")
    p.add_argument("jobs_dir", help="Job directory containing job*-output.jsonl files")
    p.add_argument("--parquet", action="store_true", help="Also write review.parquet")
    p.add_argument("--out", default=None, help="Output CSV path (default: <jobs_dir>/review.csv)")
    args = p.parse_args()

    raw      = load_raw(Path(args.input))
    analysis = load_analysis(Path(args.jobs_dir))
    merged   = join(raw, analysis)
    print_counts(raw, analysis, merged)

    out = Path(args.out) if args.out else Path(args.jobs_dir) / "review.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"CSV     → {out}")

    if args.parquet:
        pq = out.with_suffix(".parquet")
        merged.to_parquet(pq, index=False)
        print(f"Parquet → {pq}")


if __name__ == "__main__":
    main()
