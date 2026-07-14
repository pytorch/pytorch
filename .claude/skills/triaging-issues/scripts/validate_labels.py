#!/usr/bin/env python3
"""
PreToolUse hook to validate labels during triage.

This hook intercepts mcp__github__update_issue calls and blocks any attempt
to add labels that are:
1. Reserved for CI/infrastructure use (forbidden patterns)
2. Not in the labels.json allowlist (non-existent labels)

Exit codes:
  0 - Allow the tool call
  2 - Block the tool call (feedback sent to Claude)
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
SCRIPT_DIR = Path(__file__).parent
LABELS_FILE = SCRIPT_DIR.parent / "labels.json"


def debug_log(msg: str, to_stderr: bool = False):
    """Append a debug message to the log file and optionally stderr."""
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [PreToolUse] {msg}"
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if to_stderr or os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


# Patterns that match forbidden label prefixes (case-insensitive)
FORBIDDEN_PATTERNS = [
    r"^ciflow/",  # CI job triggers for PRs only
    r"^test-config/",  # Test suite selectors for PRs only
    r"^release notes:",  # Auto-assigned for release notes
    r"^ci-",  # CI infrastructure controls
    r"^ci:",  # CI infrastructure controls (includes ci: sev)
    r"^sev",  # Severity labels require human decision
    r"deprecated",  # Obsolete labels (anywhere in name)
]

# Exact label names that are forbidden (case-insensitive)
FORBIDDEN_EXACT = [
    "merge blocking",
]

# Redundant label pairs: if the specific label is present, the general label should be stripped
# Format: (specific_label, general_label_to_remove)
REDUNDANT_PAIRS = [
    ("module: rnn", "module: nn"),
]


def is_forbidden(label: str) -> bool:
    """Check if a label matches any forbidden pattern or exact name."""
    label_lower = label.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, label_lower):
            return True

    return label_lower in [f.lower() for f in FORBIDDEN_EXACT]


def load_valid_labels() -> set[str]:
    """Load the set of valid label names from labels.json.

    Raises RuntimeError if labels.json cannot be loaded, as this indicates
    a configuration problem that must be fixed.
    """
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


def find_nonexistent_labels(labels: list[str], valid_labels: set[str]) -> list[str]:
    """Return labels that don't exist in the valid labels set."""
    return [label for label in labels if label not in valid_labels]


def find_redundant_labels(labels: list[str]) -> list[tuple[str, str]]:
    """Return list of (specific, general) pairs where both are present."""
    labels_set = set(labels)
    return [
        (specific, general)
        for specific, general in REDUNDANT_PAIRS
        if specific in labels_set and general in labels_set
    ]


def main():
    try:
        data = json.load(sys.stdin)
        debug_log(f"Hook invoked with data: {json.dumps(data, indent=2)}")
        tool_input = data.get("tool_input", {})

        labels = tool_input.get("labels", []) or []
        debug_log(f"Labels to validate: {labels}")

        if not labels:
            debug_log("No labels provided, allowing")
            sys.exit(0)

        forbidden = [label for label in labels if is_forbidden(label)]
        if forbidden:
            debug_log(f"BLOCKING - forbidden labels: {forbidden}")
            print(f"BLOCKED: Cannot add forbidden labels: {forbidden}", file=sys.stderr)
            print(
                "These labels are reserved for CI/infrastructure use only.",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            print(
                "ACTION REQUIRED: Add ONLY the 'triage review' label instead.",
                file=sys.stderr,
            )
            print(
                "Do NOT add any other labels. A human will review this issue.",
                file=sys.stderr,
            )
            sys.exit(2)

        redundant = find_redundant_labels(labels)
        if redundant:
            debug_log(f"BLOCKING - redundant labels: {redundant}")
            for specific, general in redundant:
                print(
                    f"BLOCKED: Redundant labels detected: '{general}' is redundant when '{specific}' is present.",
                    file=sys.stderr,
                )
            print(file=sys.stderr)
            print(
                "ACTION REQUIRED: Remove the general label(s) and keep only the specific one(s).",
                file=sys.stderr,
            )
            generals_to_remove = [general for _, general in redundant]
            print(f"Remove: {generals_to_remove}", file=sys.stderr)
            sys.exit(2)

        valid_labels = load_valid_labels()
        nonexistent = find_nonexistent_labels(labels, valid_labels)
        if nonexistent:
            debug_log(f"BLOCKING - non-existent labels: {nonexistent}")
            print(f"BLOCKED: These labels do not exist: {nonexistent}", file=sys.stderr)
            print(
                "Labels must exist in labels.json before they can be applied.",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            print(
                "ACTION REQUIRED: Remove the non-existent labels and retry with only valid labels.",
                file=sys.stderr,
            )
            print(
                "See labels.json for the full list of available labels.",
                file=sys.stderr,
            )
            sys.exit(2)

        debug_log(f"All labels allowed: {labels}")
        sys.exit(0)

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
