"""Mark the triage target when a PreToolUse hook blocks a mutation.

A blocked mutation never reaches the tool, so the PostToolUse hook that applies
`bot-triaged` never runs and the issue is left looking untouched. This module
lets the blocking hooks leave their own marker so a blocked run stays
distinguishable from a run that never happened.
"""

import os
import subprocess
import sys
from datetime import datetime


BLOCKED_LABEL = "bot-triage-blocked"
DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
REPOSITORY_ENV = "GITHUB_REPOSITORY"
ISSUE_ENV = "TRIAGE_ISSUE_NUMBER"


def _log(message: str) -> None:
    formatted = f"[{datetime.now().isoformat()}] [BlockedMarker] {message}"
    try:
        with open(DEBUG_LOG, "a") as log:
            log.write(formatted + "\n")
    except OSError:
        pass
    if os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def mark_blocked(reason: str) -> None:
    """Best-effort label of the workflow-selected issue. Never raises.

    The target comes from the environment, never from tool_input: a mutation can
    be blocked precisely because it named a target the workflow did not choose,
    so the requested one is the one value here that cannot be trusted.
    """
    try:
        repository = os.environ.get(REPOSITORY_ENV, "")
        issue_number = os.environ.get(ISSUE_ENV, "")
        if repository.count("/") != 1 or not issue_number.isdigit():
            # Without a trusted target there is nothing safe to label.
            _log(f"No trusted target to mark ({reason}); leaving log-only")
            return

        result = subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                issue_number,
                "--repo",
                repository,
                "--add-label",
                BLOCKED_LABEL,
            ],
            capture_output=True,
            check=False,
            timeout=15,
        )
        _log(
            f"Marked {repository}#{issue_number} as blocked ({reason}); "
            f"gh exit {result.returncode}: {result.stderr.decode().strip()}"
        )
    except Exception as error:  # never let marking change the block itself
        _log(f"Could not mark blocked run: {type(error).__name__}: {error}")
