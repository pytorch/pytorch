#!/usr/bin/env python3
"""
PreToolUse hook to validate and sanitize labels during triage.

This hook intercepts mcp__github__update_issue calls and:
1. Strips forbidden labels (CI/infrastructure/severity)
2. Strips non-existent labels (not in labels.json)
3. Strips redundant labels (e.g. module: nn when module: rnn is present)
4. Fetches existing labels on the issue and merges with valid new labels
5. Rewrites the tool input via updatedInput so the MCP SET preserves existing labels

Exit codes:
  0 - Allow the tool call (with possible input rewrite via stdout JSON)
  2 - Block the tool call (feedback sent to Claude)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
SCRIPT_DIR = Path(__file__).parent
LABELS_FILE = SCRIPT_DIR.parent / "labels.json"

FORBIDDEN_PATTERNS = [
    r"^ciflow/",
    r"^test-config/",
    r"^release notes:",
    r"^ci-",
    r"^ci:",
    r"^sev",
    r"deprecated",
]

FORBIDDEN_EXACT = [
    "merge blocking",
    "oncall: releng",  # Not a triage redirect target; use module: ci instead
]

REDUNDANT_PAIRS = [
    ("module: rnn", "module: nn"),
]


def debug_log(msg: str, to_stderr: bool = False):
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [PreToolUse] {msg}"
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if to_stderr or os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def is_forbidden(label: str) -> bool:
    label_lower = label.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, label_lower):
            return True
    return label_lower in [f.lower() for f in FORBIDDEN_EXACT]


def load_valid_labels() -> set[str]:
    try:
        with open(LABELS_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"labels.json not found at {LABELS_FILE}") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"labels.json contains invalid JSON: {e}") from None
    except PermissionError:
        raise RuntimeError("Cannot read labels.json: permission denied") from None

    labels_list = data.get("labels", [])
    try:
        return {label["name"] for label in labels_list}
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"labels.json has malformed entries: {e}") from None


def strip_redundant(labels: list[str]) -> tuple[list[str], list[str]]:
    labels_set = set(labels)
    to_remove = set()
    for specific, general in REDUNDANT_PAIRS:
        if specific in labels_set and general in labels_set:
            to_remove.add(general)
    return [l for l in labels if l not in to_remove], sorted(to_remove)


def fetch_existing_labels(owner: str, repo: str, issue_number: int) -> list[str]:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--repo",
            f"{owner}/{repo}",
            "--json",
            "labels",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot fetch existing labels (gh exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    data = json.loads(result.stdout)
    return [label["name"] for label in data.get("labels", [])]


def allow_with_updated_input(tool_input: dict, merged_labels: list[str]) -> None:
    updated = dict(tool_input)
    updated["labels"] = merged_labels
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main():
    try:
        data = json.load(sys.stdin)
        debug_log(f"Hook invoked with data: {json.dumps(data, indent=2)}")
        tool_input = data.get("tool_input", {})

        requested_labels = tool_input.get("labels", []) or []
        debug_log(f"Labels requested: {requested_labels}")

        if not requested_labels:
            debug_log("No labels provided, allowing")
            sys.exit(0)

        owner = tool_input.get("owner", "pytorch")
        repo = tool_input.get("repo", "pytorch")
        issue_number = tool_input.get("issue_number")
        if not issue_number:
            raise RuntimeError("tool_input missing issue_number")

        forbidden = [l for l in requested_labels if is_forbidden(l)]
        clean_labels = [l for l in requested_labels if not is_forbidden(l)]

        if forbidden:
            debug_log(f"Stripped forbidden labels: {forbidden}")
            if not clean_labels:
                clean_labels = ["triage review"]
            elif "triage review" not in clean_labels:
                clean_labels.append("triage review")
            print(
                f"Stripped forbidden labels (require human decision): {forbidden}. "
                f"Added 'triage review' for human attention.",
                file=sys.stderr,
            )

        valid_labels = load_valid_labels()
        nonexistent = [l for l in clean_labels if l not in valid_labels]
        clean_labels = [l for l in clean_labels if l in valid_labels]

        if nonexistent:
            debug_log(f"Stripped non-existent labels: {nonexistent}")
            print(
                f"Stripped non-existent labels: {nonexistent}",
                file=sys.stderr,
            )

        clean_labels, removed_redundant = strip_redundant(clean_labels)
        if removed_redundant:
            debug_log(f"Stripped redundant labels: {removed_redundant}")
            print(
                f"Stripped redundant labels: {removed_redundant}",
                file=sys.stderr,
            )

        if not clean_labels:
            debug_log("No valid labels remain after filtering, blocking")
            print(
                "All requested labels were invalid. No labels to apply.",
                file=sys.stderr,
            )
            sys.exit(2)

        existing_labels = fetch_existing_labels(owner, repo, issue_number)
        debug_log(f"Existing labels on issue: {existing_labels}")

        merged = sorted(set(existing_labels) | set(clean_labels))
        debug_log(f"Merged labels (existing + new): {merged}")

        allow_with_updated_input(tool_input, merged)

    except json.JSONDecodeError as e:
        debug_log(f"JSON decode error: {e}")
        print(f"Hook error: Invalid JSON input: {e}", file=sys.stderr)
        print("Hook was unable to validate labels; stopping triage.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        debug_log(f"Unexpected error: {type(e).__name__}: {e}")
        print(f"Hook error: {e}", file=sys.stderr)
        print("Hook was unable to validate labels; stopping triage.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
