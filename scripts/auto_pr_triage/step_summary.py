"""Render untrusted text for GitHub step summaries and job logs."""

from __future__ import annotations

import json
from typing import Any


def summary_prose(value: str) -> str:
    """Render untrusted text as inert single-line Markdown."""

    text = " ".join(value.split())
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#"):
        text = text.replace(character, f"\\{character}")
    return text


def print_log_json(*, prefix: str, value: Any) -> None:
    """Print JSON to the job log so no line can run a workflow command.

    The runner parses `::command::` at the start of a line and the legacy
    `##[command]` anywhere in a line, so both are escaped.
    """

    for line in json.dumps(value, indent=2, sort_keys=True).splitlines():
        safe_line = line.replace("::", r"\u003a\u003a").replace("##[", r"\u0023\u0023[")
        print(f"{prefix} | {safe_line}", flush=True)
