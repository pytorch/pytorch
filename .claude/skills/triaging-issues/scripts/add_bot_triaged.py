#!/usr/bin/env python3
"""
PostToolUse hook to automatically add the bot-triaged label after any issue mutation.

This hook runs after successful GitHub issue mutations (label/comment/close/transfer)
and directly applies the `bot-triaged` label via the gh CLI.

Exit codes:
  0 - Success
"""

import json
import subprocess
import sys


BOT_TRIAGED_LABEL = "bot-triaged"


def main():
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input", {})

        owner = tool_input.get("owner")
        repo = tool_input.get("repo")
        issue_number = tool_input.get("issue_number")

        if not all([owner, repo, issue_number]):
            sys.exit(0)

        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                f"{owner}/{repo}",
                "--add-label",
                BOT_TRIAGED_LABEL,
            ],
            capture_output=True,
            check=False,
        )
        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
