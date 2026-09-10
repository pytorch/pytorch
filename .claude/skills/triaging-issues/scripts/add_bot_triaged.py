#!/usr/bin/env python3
"""
PostToolUse hook to automatically add the bot-triaged label after any issue mutation.

This hook runs after successful GitHub issue mutations (label/comment/close/transfer)
and directly applies the `bot-triaged` label via the gh CLI.

Exit codes:
  0 - Success
"""

import json
import os
import sys
from datetime import datetime

from gh_api import gh_api


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
BOT_TRIAGED_LABEL = "bot-triaged"


def debug_log(msg: str):
    """Append a debug message to the log file and stderr (for CI visibility)."""
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [PostToolUse] {msg}"
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def main():
    try:
        data = json.load(sys.stdin)
        debug_log(f"Hook invoked with data: {json.dumps(data, indent=2)}")
        tool_input = data.get("tool_input", {})

        owner = tool_input.get("owner")
        repo = tool_input.get("repo")
        issue_number = tool_input.get("issue_number")

        if not all([owner, repo, issue_number]):
            debug_log(
                f"Missing required fields - owner={owner}, repo={repo}, issue_number={issue_number}"
            )
            sys.exit(0)

        # REST append endpoint; the previous `gh issue edit` (GraphQL) path hit
        # 503s and silently left the marker off. If this still fails, the
        # catch-all below logs it and check_triage_outcome.py re-applies the
        # marker from the workflow.
        endpoint = f"repos/{owner}/{repo}/issues/{issue_number}/labels"
        gh_api(
            ["-X", "POST", endpoint, "-f", f"labels[]={BOT_TRIAGED_LABEL}"],
            log=debug_log,
        )
        sys.exit(0)

    except json.JSONDecodeError as e:
        debug_log(f"JSON decode error: {e}")
        sys.exit(0)
    except Exception as e:
        debug_log(f"Unexpected error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
