#!/usr/bin/env python3
"""
tokenizer.py — estimate token usage and cost for a batch analysis run.

Usage:
    python analysis/tokenizer.py output/f452.jsonl [--prompt analysis/prompt-1.txt]
    python analysis/tokenizer.py analysis/jobs/f452-1/job1-input.jsonl  # count actual tokens in a job

Prints a per-model comparison table and writes reports/<stem>-report.json.
Run this before openai_batch.py create.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import openai_batch as _ab

# Input pricing ($/1M tokens, batch API) — from docs/openai.md, updated 2026-05-05.
# Add Anthropic/other models here when needed; only models with a LIMITS entry are reported.
MODEL_PRICING: dict[str, float] = {
    "gpt-5.5":       2.50,
    "gpt-5.4":       1.25,
    "gpt-5.4-mini":  0.375,
    "gpt-5.4-nano":  0.10,
    "gpt-4o":        1.25,
    "gpt-4o-mini":   0.075,
    "gpt-o4-mini":   0.55,
}


def compute_report(
    comments: list[dict],
    forum: dict | None,
    prompt_hash: str,
    input_file: str,
    input_sha256: str,
    prompt_file: str,
) -> dict:
    """Compute token estimate and per-model job/cost/time breakdown."""
    # Use gpt-4o encoding as the canonical estimator (same for all current models)
    total_tokens = sum(
        _ab.estimate_tokens(_ab.build_messages(c, forum)[0], "gpt-4o")
        for c in comments
    )

    report: dict = {
        "prompt": prompt_file,
        "prompt_hash": prompt_hash,
        "input_file": input_file,
        "input_sha256": input_sha256,
        "total_comments": len(comments),
        "input_tokens": total_tokens,
    }

    for model, tpd in _ab.MODEL_LIMITS.items():
        effective_tpd = int(tpd * _ab._LIMIT_BUFFER)
        jobs = math.ceil(total_tokens / effective_tpd)
        cost = round(total_tokens / 1_000_000 * MODEL_PRICING.get(model, 0.0), 4)
        est_days = round(total_tokens / tpd, 2)
        report[model] = {"jobs": jobs, "cost_$": cost, "est_queue_days": est_days}

    return report


def count_input_tokens(path: Path, model: str = "gpt-4o") -> dict:
    """Count tokens in an existing job input JSONL (batch request format).

    Each line must have body.messages (as written by build_batch_request_line).
    Returns {"total_tokens": int, "total_requests": int, "min": int, "max": int, "mean": float}.
    """
    counts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            messages = req["body"]["messages"]
            counts.append(_ab.estimate_tokens(messages, model))
    if not counts:
        return {"total_tokens": 0, "total_requests": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "total_tokens": sum(counts),
        "total_requests": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 1),
    }


def print_table(report: dict) -> None:
    """Print a human-readable model comparison table to stdout."""
    print(f"\nInput:    {report['input_file']}")
    print(f"Comments: {report['total_comments']:,}")
    print(f"Tokens:   {report['input_tokens']:,}")
    print(f"Prompt:   {report['prompt']}  (hash: {report['prompt_hash']})")
    print()

    # Cheapest model that fits in one job
    single_job_models = [m for m in _ab.MODEL_LIMITS if report.get(m, {}).get("jobs") == 1]
    best = (min(single_job_models, key=lambda m: report[m]["cost_$"])
            if single_job_models else None)

    print(f"{'Model':<15} {'Jobs':>5}  {'Cost ($)':>9}  {'Est days':>9}  {'Note'}")
    print("-" * 62)
    for model in _ab.MODEL_LIMITS:
        if model not in report or not isinstance(report[model], dict):
            continue
        m = report[model]
        note = "<-- recommended" if model == best else ""
        print(f"{model:<15} {m['jobs']:>5}  {m['cost_$']:>9.4f}  {m['est_queue_days']:>9.2f}  {note}")
    print()


def _is_job_input(path: Path) -> bool:
    """Return True if this JSONL looks like a batch request file (has custom_id)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return "custom_id" in json.loads(line)
    return False


def main() -> None:
    _default_prompt = Path(__file__).parent / "prompt-1.txt"

    parser = argparse.ArgumentParser(description="Estimate batch token usage and cost.")
    parser.add_argument("input", help="Scraped JSONL or job input JSONL (jobN-input.jsonl)")
    parser.add_argument(
        "--prompt",
        default=str(_default_prompt),
        help=f"System prompt file (default: {_default_prompt.name})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    # --- Mode: count tokens in an existing job input file ---
    if _is_job_input(input_path):
        result = count_input_tokens(input_path)
        print(f"\nJob input: {input_path.name}")
        print(f"  Requests : {result['total_requests']:,}")
        print(f"  Tokens   : {result['total_tokens']:,}")
        print(f"  Per-req  : min={result['min']}  max={result['max']}  mean={result['mean']}")
        return

    # --- Mode: estimate from raw scrape file and write report.json ---
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        sys.exit(f"Prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:7]

    # Ensure build_messages uses the specified prompt
    _ab._load_prompt(prompt_path)

    forum, comments = _ab.load_items(input_path)
    if not comments:
        sys.exit("No comment items found.")
    if forum is None:
        print("Warning: no ForumItem — token estimates may be slightly low.", file=sys.stderr)

    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    report = compute_report(
        comments, forum, prompt_hash,
        str(input_path), input_sha256, str(prompt_path),
    )

    print_table(report)

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"{input_path.stem}-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to: {out_path}")
    print(f"\nNext:  python analysis/openai_batch.py create {out_path} --model <model>")


if __name__ == "__main__":
    main()
