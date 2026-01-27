#!/usr/bin/env python3
"""
PreToolUse hook to block forbidden labels from being added during triage.

This hook intercepts mcp__github__update_issue calls and blocks any attempt
to add labels that are reserved for CI/infrastructure use.

Exit codes:
  0 - Allow the tool call
  2 - Block the tool call (feedback sent to Claude)
"""

import json
import re
import sys


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


def is_forbidden(label: str) -> bool:
    """Check if a label matches any forbidden pattern or exact name."""
    label_lower = label.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, label_lower):
            return True

    return label_lower in [f.lower() for f in FORBIDDEN_EXACT]


def main():
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input", {})

        # Handle both possible field names for labels
        labels = tool_input.get("labels", []) or []

        # If no labels being set, allow
        if not labels:
            sys.exit(0)

        # Check each label
        blocked = [label for label in labels if is_forbidden(label)]

        if blocked:
            # Exit code 2 blocks the tool and sends stderr as feedback to Claude
            print(f"BLOCKED: Cannot add forbidden labels: {blocked}", file=sys.stderr)
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

        sys.exit(0)

    except json.JSONDecodeError as e:
        print(f"Hook error: Invalid JSON input: {e}", file=sys.stderr)
        print("Hook was unable to validate labels; stopping triage.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Hook error: {e}", file=sys.stderr)
        print("Hook was unable to validate labels; stopping triage.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
