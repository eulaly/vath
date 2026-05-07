#!/usr/bin/env python3
"""
openai_batch.py — OpenAI Batch API job runner

Run tokenizer.py first to generate report.json, then:
    create   <report.json> --model <model>   — build job directory
    submit   [--job N] [--dir DIR]           — submit next eligible job
    status   [--job N] [--dir DIR]           — check job status
    download [--job N] [--dir DIR]           — download + normalize completed jobs

DIR is a name under analysis/jobs/ (default: most recently created).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    sys.exit("openai package not installed. Run: pip install openai")

# ---------------------------------------------------------------------------
# Model limits and token estimation

# Max enqueued tokens across ALL concurrent batches (docs/openai.md, 2026-05-05).
# Org-tier limits may be lower; use --job to limit submission size if needed.
MODEL_LIMITS: dict[str, int] = {
    "gpt-5.5":        900_000,
    "gpt-5.4":        900_000,
    "gpt-5.4-mini": 2_000_000,
    "gpt-5.4-nano":   200_000,
    "gpt-4o":         900_000,
    "gpt-4o-mini":  2_000_000,
    "gpt-o4-mini":  2_000_000,
}
_DEFAULT_TOKEN_LIMIT = 900_000
_MODEL_ENCODING: dict[str, str] = {
    "gpt-5.5":       "o200k_base",
    "gpt-5.4":       "o200k_base",
    "gpt-5.4-mini":  "o200k_base",
    "gpt-5.4-nano":  "o200k_base",
    "gpt-4o":        "o200k_base",
    "gpt-4o-mini":   "o200k_base",
    "gpt-o4-mini":   "o200k_base",
}
_LIMIT_BUFFER = 0.80


def estimate_tokens(messages: list[dict], model: str) -> int:
    """Token count per OpenAI cookbook chat formula; falls back to chars/3."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(_MODEL_ENCODING.get(model, "o200k_base"))
        # Per OpenAI cookbook for gpt-4o: 3 overhead per message + role + content;
        # plus 3 tokens for the reply primer (<|start|>assistant<|message|>).
        total = 3  # reply primer
        for m in messages:
            total += 3
            total += len(enc.encode(m.get("role", "")))
            total += len(enc.encode(m["content"]))
        return total
    except ImportError:
        return 3 + sum(3 + len(m["content"]) // 3 for m in messages)


def chunk_comments_by_tokens(
    comments: list[dict], forum: dict | None, model: str
) -> list[list[dict]]:
    """Greedy bin-pack comments into chunks that fit under the model TPD limit."""
    token_limit = int(MODEL_LIMITS.get(model, _DEFAULT_TOKEN_LIMIT) * _LIMIT_BUFFER)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for comment in comments:
        messages, _ = build_messages(comment, forum)
        tokens = estimate_tokens(messages, model)
        if current and current_tokens + tokens > token_limit:
            chunks.append(current)
            current = [comment]
            current_tokens = tokens
        else:
            current.append(comment)
            current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Prompt

_DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompt-1.txt"
SYSTEM_PROMPT = _DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:7]


def _load_prompt(path: Path) -> None:
    global SYSTEM_PROMPT, PROMPT_VERSION
    SYSTEM_PROMPT = path.read_text(encoding="utf-8").strip()
    PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:7]


USER_TEMPLATE = """\
## Proposed Regulation
Title: {reg_title}
Description: {reg_desc}

---

## Public Comment
Comment ID: {comment_id}
Title: {comment_title}
Body:
{comment_text}

---
Classify this comment per the instructions. Return only JSON.\
"""

MAX_COMMENT_CHARS = 6000

# ---------------------------------------------------------------------------
# Directories

_SCRIPT_DIR = Path(__file__).parent
JOBS_DIR = _SCRIPT_DIR / "jobs"

# ---------------------------------------------------------------------------
# Core functions (importable for tests)


def load_items(path: Path) -> tuple[dict | None, list[dict]]:
    """Read a scraped JSONL. Returns (forum_item_or_None, [comment_items])."""
    forum = None
    comments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "comment_id" in item:
                comments.append(item)
            elif "reg_title" in item:
                forum = item
    return forum, comments


def custom_id_from(comment_id: str) -> str:
    return f"comment_{comment_id}"


def parse_custom_id(custom_id: str) -> str:
    return custom_id.removeprefix("comment_")


def build_messages(comment: dict, forum: dict | None) -> tuple[list, bool]:
    """Build OpenAI messages for one comment. Returns (messages, truncated)."""
    reg_title = (forum or {}).get("reg_title", "[unknown]")
    reg_desc  = (forum or {}).get("reg_desc",  "[unknown]")
    body = (comment.get("text") or "").strip()
    truncated = False
    if not body:
        body = "[No body text provided]"
    elif len(body) > MAX_COMMENT_CHARS:
        body = body[:MAX_COMMENT_CHARS] + "... [truncated]"
        truncated = True
    user_text = USER_TEMPLATE.format(
        reg_title=reg_title,
        reg_desc=reg_desc,
        comment_id=comment.get("comment_id", ""),
        comment_title=comment.get("title", ""),
        comment_text=body,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ], truncated


def build_batch_request_line(comment: dict, forum: dict | None, model: str) -> dict:
    messages, _ = build_messages(comment, forum)
    return {
        "custom_id": custom_id_from(comment["comment_id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        },
    }


def normalize_output_line(
    raw_line: dict,
    comment_lookup: dict,
    run_id: str,
    analyzed_at: str,
    model: str,
    prompt_version: str,
) -> dict:
    """Convert one raw batch output line into a normalized analysis record."""
    comment_id = parse_custom_id(raw_line.get("custom_id", ""))
    comment = comment_lookup.get(comment_id, {})
    base = {
        "run_id":         run_id,
        "forum_id":       comment.get("forum_id", ""),
        "comment_id":     comment_id,
        "analyzed_at":    analyzed_at,
        "model":          model,
        "prompt_version": prompt_version,
        "input_title":    comment.get("title", ""),
        "truncated":      len(comment.get("text") or "") > MAX_COMMENT_CHARS,
    }
    if raw_line.get("error"):
        err = raw_line["error"]
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return {**base, "stance": None, "stance_confidence": None,
                "stance_rationale": None, "tone": None, "tags": None, "error": err_msg}
    response = raw_line.get("response") or {}
    if response.get("status_code") != 200:
        return {**base, "stance": None, "stance_confidence": None,
                "stance_rationale": None, "tone": None, "tags": None,
                "error": f"status {response.get('status_code')}"}
    try:
        content = response["body"]["choices"][0]["message"]["content"]
        data = json.loads(content)
        keys = ("stance", "stance_confidence", "stance_rationale", "tone", "tags")
        parsed = {k: data.get(k) for k in keys}
        return {**base, **parsed, "error": None}
    except Exception as exc:
        return {**base, "stance": None, "stance_confidence": None,
                "stance_rationale": None, "tone": None, "tags": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Job directory management


def _next_job_dir(stem: str) -> Path:
    base = stem[:8]
    i = 1
    while (JOBS_DIR / f"{base}-{i}").exists():
        i += 1
    return JOBS_DIR / f"{base}-{i}"


def _latest_job_dir() -> Path:
    if not JOBS_DIR.exists():
        sys.exit(f"No jobs directory found. Run 'create' first.")
    status_files = list(JOBS_DIR.glob("*/status.json"))
    if not status_files:
        sys.exit(f"No jobs found in {JOBS_DIR}. Run 'create' first.")
    return max(status_files, key=lambda p: p.stat().st_mtime).parent


def _resolve_job_dir(args) -> Path:
    if getattr(args, "dir", None):
        d = Path(args.dir)
        if not d.is_absolute():
            d = JOBS_DIR / d
        if not d.exists():
            sys.exit(f"Job directory not found: {d}")
        return d
    return _latest_job_dir()


def load_status(job_dir: Path) -> dict:
    return json.loads((job_dir / "status.json").read_text(encoding="utf-8"))


def save_status(status: dict, job_dir: Path) -> None:
    (job_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _find_next_eligible_job(jobs: list[dict]) -> tuple[dict | None, str | None]:
    """Return (next_pending_job, None) or (None, warning_message).

    A job is eligible when it is 'pending' and either it is the first job
    or its predecessor is 'completed'.
    """
    for j in jobs:
        if j["status"] != "pending":
            continue
        if j["job_num"] == 1:
            return j, None
        prev = next(p for p in jobs if p["job_num"] == j["job_num"] - 1)
        if prev["status"] == "completed":
            return j, None
        if prev["status"] in ("submitted", "in_progress", "validating", "finalizing"):
            return None, (
                f"Job {prev['job_num']} is '{prev['status']}'. "
                f"Wait for it to complete before submitting job {j['job_num']}."
            )
    return None, None


# ---------------------------------------------------------------------------
# Subcommand: create


def cmd_create(args) -> None:
    report_path = Path(args.report)
    if not report_path.exists():
        sys.exit(f"Report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))

    if args.model not in report or not isinstance(report[args.model], dict):
        available = [k for k in report if isinstance(report.get(k), dict)]
        sys.exit(f"Model '{args.model}' not in report. Available: {', '.join(available)}")

    prompt_path = Path(report["prompt"])
    if not prompt_path.exists():
        sys.exit(f"Prompt file not found: {prompt_path}")
    _load_prompt(prompt_path)

    input_path = Path(report["input_file"])
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    forum, comments = load_items(input_path)
    if not comments:
        sys.exit("No comment items found in input file.")

    chunks = chunk_comments_by_tokens(comments, forum, args.model)

    stem = input_path.stem[:8]
    job_dir = _next_job_dir(stem)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir()

    shutil.copy2(input_path, job_dir / "forum.jsonl")
    shutil.copy2(prompt_path, job_dir / "prompt.txt")
    shutil.copy2(report_path, job_dir / "report.json")

    jobs_meta = []
    for i, chunk in enumerate(chunks, start=1):
        req_path = job_dir / f"job{i}-input.jsonl"
        with open(req_path, "w", encoding="utf-8") as f:
            for comment in chunk:
                f.write(json.dumps(build_batch_request_line(comment, forum, args.model),
                                   ensure_ascii=False) + "\n")
        jobs_meta.append({
            "job_num": i,
            "run_id": str(uuid.uuid4()),
            "status": "pending",
            "batch_id": None,
            "records_submitted": len(chunk),
            "records_completed": None,
            "records_failed": None,
            "submitted_at": None,
            "completed_at": None,
        })

    model_info = report[args.model]
    status = {
        "model": args.model,
        "prompt_hash": report["prompt_hash"],
        "input_file": str(input_path),
        "input_sha256": report["input_sha256"],
        "total_comments": report["total_comments"],
        "input_tokens": report["input_tokens"],
        "est_queue_days": model_info["est_queue_days"],
        "cost_$": model_info["cost_$"],
        "total_jobs": len(chunks),
        "jobs": jobs_meta,
    }
    save_status(status, job_dir)

    print(f"Created: {job_dir.name}")
    print(f"  {len(chunks)} job(s)  |  {len(comments)} comments  |  model: {args.model}")
    print(f"\nNext:  python analysis/openai_batch.py submit")


# ---------------------------------------------------------------------------
# Subcommand: submit


def cmd_submit(args, client) -> None:
    job_dir = _resolve_job_dir(args)
    status = load_status(job_dir)
    jobs = status["jobs"]

    if args.job:
        target = next((j for j in jobs if j["job_num"] == args.job), None)
        if target is None:
            sys.exit(f"Job {args.job} not found in {job_dir.name}.")
        if target["status"] != "pending":
            sys.exit(f"Job {args.job} is already '{target['status']}' — cannot resubmit.")
        if target["job_num"] > 1:
            prev = next(p for p in jobs if p["job_num"] == target["job_num"] - 1)
            if prev["status"] != "completed":
                sys.exit(
                    f"Cannot submit job {target['job_num']}: "
                    f"job {prev['job_num']} is '{prev['status']}' (must be 'completed')."
                )
    else:
        target, warning = _find_next_eligible_job(jobs)
        if warning:
            print(warning, file=sys.stderr)
            sys.exit(1)
        if target is None:
            all_done = all(j["status"] == "completed" for j in jobs)
            print("All jobs completed." if all_done else "No pending jobs eligible for submission.")
            return

    n = target["job_num"]
    req_path = job_dir / f"job{n}-input.jsonl"
    print(f"Submitting job {n}/{status['total_jobs']} ({target['records_submitted']} comments) ...",
          file=sys.stderr)

    with open(req_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"run_id": target["run_id"], "job_dir": job_dir.name},
    )

    target["status"] = "submitted"
    target["batch_id"] = batch.id
    target["submitted_at"] = datetime.now(timezone.utc).isoformat()
    save_status(status, job_dir)

    print(f"Job {n} submitted: {batch.id}  ({batch.status})")
    print(f"  python analysis/openai_batch.py status")


# ---------------------------------------------------------------------------
# Subcommand: status


def cmd_status(args, client) -> None:
    job_dir = _resolve_job_dir(args)
    status = load_status(job_dir)
    jobs = status["jobs"]

    job_filter = getattr(args, "job", None)

    for job in jobs:
        if job_filter is not None and job["job_num"] != job_filter:
            continue
        if not job["batch_id"]:
            continue
        if job["status"] in ("completed", "failed", "expired", "cancelled", "pending"):
            continue
        batch = client.batches.retrieve(job["batch_id"])
        counts = batch.request_counts
        if batch.status == "completed":
            job["status"] = "completed"
            if batch.completed_at:
                job["completed_at"] = datetime.fromtimestamp(
                    batch.completed_at, tz=timezone.utc
                ).isoformat()
        elif batch.status in ("failed", "expired", "cancelled"):
            job["status"] = batch.status
        else:
            job["status"] = batch.status
        job["records_completed"] = counts.completed
        job["records_failed"] = counts.failed

    save_status(status, job_dir)

    target_jobs = jobs if not job_filter else [j for j in jobs if j["job_num"] == job_filter]
    print(f"Dir: {job_dir.name}  |  Model: {status['model']}  |  {status['total_jobs']} job(s)")
    print(f"{'Job':<5} {'Status':<14} {'Records':>12}  {'Submitted':<20}  {'Completed':<20}")
    print("-" * 76)
    for j in target_jobs:
        rec = (f"{j['records_completed']}/{j['records_submitted']}"
               if j["records_completed"] is not None else f"-/{j['records_submitted']}")
        sub  = (j["submitted_at"]  or "-")[:19]
        done = (j["completed_at"] or "-")[:19]
        print(f"{j['job_num']:<5} {j['status']:<14} {rec:>12}  {sub:<20}  {done:<20}")


# ---------------------------------------------------------------------------
# Subcommand: download


def cmd_download(args, client) -> None:
    job_dir = _resolve_job_dir(args)

    # Refresh status before deciding what to download
    cmd_status(args, client)
    status = load_status(job_dir)
    jobs = status["jobs"]

    job_filter = getattr(args, "job", None)
    if job_filter:
        candidates = [j for j in jobs if j["job_num"] == job_filter]
    else:
        candidates = [
            j for j in jobs
            if j["status"] == "completed"
            and not (job_dir / f"job{j['job_num']}-output.jsonl").exists()
        ]

    if not candidates:
        print("No completed jobs pending download.", file=sys.stderr)
        return

    _, all_comments = load_items(job_dir / "forum.jsonl")
    comment_lookup = {c["comment_id"]: c for c in all_comments}

    for job in candidates:
        n = job["job_num"]

        if job["status"] != "completed":
            print(f"Job {n} not yet completed ('{job['status']}'), skipping.", file=sys.stderr)
            continue

        batch = client.batches.retrieve(job["batch_id"])

        if not batch.output_file_id:
            print(f"Job {n}: no output file available from OpenAI.", file=sys.stderr)
            continue

        raw_text = client.files.content(batch.output_file_id).text
        raw_path = job_dir / f"job{n}-output-raw.jsonl"
        raw_path.write_text(raw_text, encoding="utf-8")
        print(f"Job {n} raw → {raw_path.name}", file=sys.stderr)

        if batch.error_file_id:
            err_text = client.files.content(batch.error_file_id).text
            err_path = job_dir / f"job{n}-errors.jsonl"
            err_path.write_text(err_text, encoding="utf-8")
            n_err_lines = sum(1 for line in err_text.splitlines() if line.strip())
            print(f"Job {n} errors → {err_path.name}  ({n_err_lines} lines)", file=sys.stderr)

        completed_at = job.get("completed_at") or datetime.now(timezone.utc).isoformat()
        norm_path = job_dir / f"job{n}-output.jsonl"
        n_ok = n_err = 0
        with open(norm_path, "w", encoding="utf-8") as out:
            for line in raw_text.splitlines():
                if not line.strip():
                    continue
                record = normalize_output_line(
                    json.loads(line), comment_lookup,
                    job["run_id"], completed_at,
                    status["model"], status["prompt_hash"],
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if record["error"]:
                    n_err += 1
                else:
                    n_ok += 1

        print(f"Job {n} normalized → {norm_path.name}  ({n_ok} ok, {n_err} errors)", file=sys.stderr)
        job["records_completed"] = n_ok
        job["records_failed"] = n_err

    save_status(status, job_dir)


# ---------------------------------------------------------------------------
# CLI


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--job", type=int, default=None, metavar="N",
                   help="Job number within the run (default: auto)")
    p.add_argument("--dir", default=None, metavar="DIR",
                   help="Job directory name or path (default: most recent)")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Batch analysis job runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create job directory from tokenizer report")
    p_create.add_argument("report", help="Path to report.json from tokenizer.py")
    p_create.add_argument("--model", required=True, help="Model (e.g. gpt-4o-mini)")

    p_submit = sub.add_parser("submit", help="Submit next eligible job")
    _add_common_args(p_submit)

    p_status = sub.add_parser("status", help="Check job status")
    _add_common_args(p_status)

    p_download = sub.add_parser("download", help="Download and normalize completed jobs")
    _add_common_args(p_download)

    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set. Create a .env file or export the variable.")
    client = openai.OpenAI(api_key=api_key)

    if args.command == "submit":
        cmd_submit(args, client)
    elif args.command == "status":
        cmd_status(args, client)
    elif args.command == "download":
        cmd_download(args, client)


if __name__ == "__main__":
    main()
