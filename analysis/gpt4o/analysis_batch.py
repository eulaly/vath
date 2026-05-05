#!/usr/bin/env python3
"""
analysis/gpt4o/analysis-batch.py — OpenAI Batch API pipeline

Commands (run manually in order):
    submit   <input_jsonl> [--model gpt-4o]   — build request file, upload, create batch
    status   <run_id>                          — check batch status, update manifest
    download <run_id>                          — download + normalize output, update manifest

File layout (all under analysis/gpt4o/):
    requests/<run_id>.jsonl     — batch input sent to OpenAI
    raw/<run_id>.jsonl          — raw batch output from OpenAI
    runs/<run_id>.json          — run manifest
    <run_id>_<model>.jsonl      — normalized output (same schema as realtime)
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    sys.exit("openai package not installed. Run: pip install openai")

# ---------------------------------------------------------------------------
# Prompt

_DEFAULT_PROMPT_FILE = Path(__file__).parent.parent / "prompt-1.txt"
SYSTEM_PROMPT = _DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:7]


def _load_prompt(path: Path) -> None:
    """Re-read a prompt file, updating module-level SYSTEM_PROMPT and PROMPT_VERSION."""
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

_SCRIPT_DIR  = Path(__file__).parent
REQUESTS_DIR = _SCRIPT_DIR / "requests"
RAW_DIR      = _SCRIPT_DIR / "raw"
RUNS_DIR     = _SCRIPT_DIR / "runs"

# ---------------------------------------------------------------------------
# Core functions (importable for tests)


def load_items(path: Path) -> tuple[dict | None, list[dict]]:
    """Read a scraped JSONL file. Returns (forum_item_or_None, [comment_items])."""
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
    """Return comment_id from a custom_id string."""
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
    """Build one line of the batch input JSONL."""
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
    """Convert one raw batch output line into a normalized analysis record.

    comment_lookup: {comment_id: CommentItem dict}
    prompt_version: taken from the run manifest so it reflects what was submitted.
    """
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

    # Check for outer-level batch error (e.g. batch_expired)
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


def make_manifest(
    run_id: str,
    input_filename: str,
    input_sha256: str,
    model: str,
    batch_id: str,
    records_submitted: int,
    request_filename: str,
) -> dict:
    return {
        "run_id":                 run_id,
        "input_filename":         input_filename,
        "input_sha256":           input_sha256,
        "prompt_hash":            PROMPT_VERSION,
        "model":                  model,
        "batch_id":               batch_id,
        "records_submitted":      records_submitted,
        "records_completed":      None,
        "records_failed":         None,
        "request_filename":       request_filename,
        "raw_output_filename":    None,
        "normalized_output_filename": None,
        "created_at":             datetime.now(timezone.utc).isoformat(),
        "completed_at":           None,
    }


def load_manifest(run_id: str) -> dict:
    path = RUNS_DIR / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand: submit

def cmd_submit(args, client) -> None:
    _load_prompt(Path(args.prompt))
    print(f"Prompt: {args.prompt}  (version {PROMPT_VERSION})", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    print(f"Reading {input_path} ...", file=sys.stderr)
    forum, comments = load_items(input_path)
    if not comments:
        sys.exit("No comment items found in input file.")
    if forum is None:
        print("Warning: no ForumItem found — regulation context will be [unknown].", file=sys.stderr)

    import uuid
    run_id = str(uuid.uuid4())
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    # Build batch request file
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    request_path = REQUESTS_DIR / f"{run_id}.jsonl"
    with open(request_path, "w", encoding="utf-8") as f:
        for comment in comments:
            line = build_batch_request_line(comment, forum, args.model)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(comments)} requests → {request_path}", file=sys.stderr)

    # Upload to OpenAI
    print("Uploading request file ...", file=sys.stderr)
    with open(request_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: {uploaded.id}", file=sys.stderr)

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"run_id": run_id, "input_filename": str(input_path)},
    )
    print(f"Batch created: {batch.id}  status={batch.status}", file=sys.stderr)

    # Save manifest
    manifest = make_manifest(
        run_id=run_id,
        input_filename=str(input_path),
        input_sha256=input_sha256,
        model=args.model,
        batch_id=batch.id,
        records_submitted=len(comments),
        request_filename=str(request_path),
    )
    save_manifest(manifest)

    print(f"\nrun_id: {run_id}", file=sys.stderr)
    print(f"Check status:   python analysis/gpt4o/analysis-batch.py status {run_id}", file=sys.stderr)
    print(run_id)  # stdout for scripting


# ---------------------------------------------------------------------------
# Subcommand: status

def cmd_status(args, client) -> None:
    manifest = load_manifest(args.run_id)
    batch = client.batches.retrieve(manifest["batch_id"])

    counts = batch.request_counts
    print(f"status:     {batch.status}")
    print(f"completed:  {counts.completed}/{counts.total}")
    print(f"failed:     {counts.failed}")

    manifest["records_completed"] = counts.completed
    manifest["records_failed"]    = counts.failed
    save_manifest(manifest)

    if batch.status == "completed":
        print(f"\nReady to download. Run:")
        print(f"  python analysis/gpt4o/analysis-batch.py download {args.run_id}")


# ---------------------------------------------------------------------------
# Subcommand: download

def cmd_download(args, client) -> None:
    manifest = load_manifest(args.run_id)
    batch = client.batches.retrieve(manifest["batch_id"])

    if batch.status != "completed":
        sys.exit(f"Batch not complete yet (status={batch.status}). Run 'status' to check.")

    run_id    = manifest["run_id"]
    model     = manifest["model"]
    model_slug = model.replace("/", "-")

    # Download raw output
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{run_id}.jsonl"
    raw_text = client.files.content(batch.output_file_id).text
    raw_path.write_text(raw_text, encoding="utf-8")
    print(f"Raw output → {raw_path}", file=sys.stderr)

    # Build comment lookup from original input for reconciliation
    input_path = Path(manifest["input_filename"])
    _, comments = load_items(input_path)
    comment_lookup = {c["comment_id"]: c for c in comments}

    # Normalize
    completed_at = datetime.now(timezone.utc).isoformat()
    if batch.completed_at:
        completed_at = datetime.fromtimestamp(batch.completed_at, tz=timezone.utc).isoformat()

    normalized_path = _SCRIPT_DIR / f"{run_id}_{model_slug}.jsonl"
    n_ok = n_err = 0
    with open(normalized_path, "w", encoding="utf-8") as out:
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            raw_line = json.loads(line)
            record = normalize_output_line(raw_line, comment_lookup, run_id, completed_at, model, manifest["prompt_hash"])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["error"]:
                n_err += 1
            else:
                n_ok += 1

    print(f"Normalized → {normalized_path}  ({n_ok} ok, {n_err} errors)", file=sys.stderr)

    manifest["records_completed"]         = n_ok
    manifest["records_failed"]            = n_err
    manifest["raw_output_filename"]       = str(raw_path)
    manifest["normalized_output_filename"] = str(normalized_path)
    manifest["completed_at"]              = completed_at
    save_manifest(manifest)
    print(f"Manifest updated → {RUNS_DIR / run_id}.json", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set. Create a .env file or export the variable.")

    parser = argparse.ArgumentParser(
        description="Public comment batch analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Build and submit a batch job")
    p_submit.add_argument("input", help="Path to scraped JSONL file")
    p_submit.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    p_submit.add_argument(
        "--prompt",
        default=str(_DEFAULT_PROMPT_FILE),
        help="Path to system prompt file (default: analysis/prompt-1.txt)",
    )

    p_status = sub.add_parser("status", help="Check batch status")
    p_status.add_argument("run_id", help="run_id from submit output")

    p_download = sub.add_parser("download", help="Download and normalize completed batch")
    p_download.add_argument("run_id", help="run_id from submit output")

    args = parser.parse_args()
    client = openai.OpenAI(api_key=api_key)

    if args.command == "submit":
        cmd_submit(args, client)
    elif args.command == "status":
        cmd_status(args, client)
    elif args.command == "download":
        cmd_download(args, client)


if __name__ == "__main__":
    main()
