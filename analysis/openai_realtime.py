#!/usr/bin/env python3
"""
analysis/openai_realtime.py — Synchronous GPT-4o pipeline for VA Townhall comments.

Usage:
    python analysis/openai_realtime.py <input_jsonl> [--limit {5,10,20,50}] [--model MODEL]

Output:
    analysis/forum{id}_{scrape_ts}_{model}_{run_ts}.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    sys.exit("openai package not installed. Run: pip install openai")

# ---------------------------------------------------------------------------
# Prompt — loaded from analysis/prompt-1.txt at import time

_PROMPT_FILE = Path(__file__).parent / "prompt-1.txt"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8").strip()
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
_RETRY_DELAYS = [1.0, 2.0]

# ---------------------------------------------------------------------------
# Core functions


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


def parse_api_response(content: str) -> dict:
    data = json.loads(content)
    keys = ("stance", "stance_confidence", "stance_rationale", "tone", "tags")
    return {k: data.get(k) for k in keys}


def _call_api(client, messages: list, model: str) -> str:
    last_exc = None
    for delay in [0.0] + _RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def analyze_comment(client, comment: dict, forum: dict | None, run_id: str, model: str) -> dict:
    base = {
        "run_id":         run_id,
        "forum_id":       comment.get("forum_id", ""),
        "comment_id":     comment.get("comment_id", ""),
        "analyzed_at":    datetime.now(timezone.utc).isoformat(),
        "model":          model,
        "prompt_version": PROMPT_VERSION,
        "input_title":    comment.get("title", ""),
    }
    try:
        messages, truncated = build_messages(comment, forum)
        content = _call_api(client, messages, model)
        parsed = parse_api_response(content)
        return {**base, **parsed, "truncated": truncated, "error": None}
    except Exception as exc:
        return {
            **base,
            "stance": None, "stance_confidence": None,
            "stance_rationale": None, "tone": None, "tags": None,
            "truncated": False,
            "error": str(exc),
        }


def _scrape_ts_from_filename(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d\-+:]+)", path.stem)
    return m.group(1).replace(":", "-") if m else "unknown"


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Analyze VA Townhall public comments with GPT-4o (synchronous).",
    )
    parser.add_argument("input", help="Path to scraped JSONL file")
    parser.add_argument(
        "--limit",
        type=int,
        choices=[5, 10, 20, 50],
        metavar="{5,10,20,50}",
        help="Process only the first N comments (for testing). Omit to process all.",
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set. Create a .env file or export the variable.")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    print(f"Reading {input_path} ...", file=sys.stderr)
    forum, comments = load_items(input_path)

    if forum is None:
        print("Warning: no ForumItem found — regulation context will be [unknown].", file=sys.stderr)

    if args.limit:
        comments = comments[: args.limit]

    forum_id   = (forum or {}).get("forum_id", "unknown")
    scrape_ts  = _scrape_ts_from_filename(input_path)
    run_ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S+00-00")
    model_slug = args.model.replace("/", "-")

    out_dir  = Path(__file__).parent
    out_path = out_dir / f"forum{forum_id}_{scrape_ts}_{model_slug}_{run_ts}.jsonl"

    run_id = str(uuid.uuid4())
    client = openai.OpenAI(api_key=api_key)

    n_ok = n_err = 0
    total = len(comments)
    print(f"Analyzing {total} comments → {out_path}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, comment in enumerate(comments, 1):
            record = analyze_comment(client, comment, forum, run_id, args.model)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if record["error"]:
                n_err += 1
                print(f"  [{i}/{total}] ERROR {comment.get('comment_id')}: {record['error']}", file=sys.stderr)
            else:
                n_ok += 1
                print(f"  [{i}/{total}] OK    {comment.get('comment_id')} → {record['stance']}", file=sys.stderr)
            time.sleep(0.1)

    print(f"\nDone. {n_ok} ok, {n_err} errors → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
