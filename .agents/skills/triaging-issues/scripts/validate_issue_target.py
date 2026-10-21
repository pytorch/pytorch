#!/usr/bin/env python3
"""Block issue mutations outside the workflow's trusted triage target."""

import json
import os
import sys
from datetime import datetime


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
REPOSITORY_ENV = "GITHUB_REPOSITORY"
ISSUE_ENV = "TRIAGE_ISSUE_NUMBER"


def debug_log(message: str) -> None:
    """Record target-validation decisions for workflow diagnostics."""
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [TargetPreToolUse] {message}"
    try:
        with open(DEBUG_LOG, "a") as log:
            log.write(formatted + "\n")
    except OSError:
        pass
    if os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def expected_target() -> tuple[str, str, int]:
    """Load and validate the target supplied by the trusted workflow."""
    repository = os.environ.get(REPOSITORY_ENV, "")
    issue_number = os.environ.get(ISSUE_ENV, "")

    if repository.count("/") != 1:
        raise RuntimeError(f"{REPOSITORY_ENV} must be in owner/repo form")
    if not issue_number.isdigit() or int(issue_number) <= 0:
        raise RuntimeError(f"{ISSUE_ENV} must be a positive integer")

    owner, repo = repository.split("/", 1)
    if not owner or not repo:
        raise RuntimeError(f"{REPOSITORY_ENV} must be in owner/repo form")
    return owner, repo, int(issue_number)


def requested_target(tool_input: dict) -> tuple[str, str, int]:
    """Extract the repository and issue targeted by an MCP mutation."""
    owner = tool_input.get("owner")
    repo = tool_input.get("repo")
    issue_number = tool_input.get("issue_number")

    if not isinstance(owner, str) or not owner:
        raise RuntimeError("tool_input missing owner")
    if not isinstance(repo, str) or not repo:
        raise RuntimeError("tool_input missing repo")
    if isinstance(issue_number, bool) or not str(issue_number).isdigit():
        raise RuntimeError("tool_input missing valid issue_number")
    return owner, repo, int(issue_number)


def main() -> None:
    """Allow only mutations aimed at the workflow-selected issue."""
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input", {})
        if (
            data.get("tool_name") == "mcp__github__issue_write"
            and tool_input.get("method") != "update"
        ):
            raise RuntimeError(
                "issue_write is restricted to updating the triage target"
            )

        expected = expected_target()
        requested = requested_target(tool_input)
        debug_log(f"Expected target {expected}; requested target {requested}")

        if requested != expected:
            raise RuntimeError(
                f"mutation target {requested[0]}/{requested[1]}#{requested[2]} "
                f"does not match triage target {expected[0]}/{expected[1]}#{expected[2]}"
            )
    except Exception as error:
        debug_log(f"Blocked mutation: {type(error).__name__}: {error}")
        print(f"Blocked issue mutation: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
