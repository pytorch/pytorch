"""Render untrusted text for GitHub step summaries and job logs."""

from __future__ import annotations

import json
import re
from typing import Any


# Characters that could close a <code> element, start Markdown, or split a
# table cell, written as entities so untrusted text renders literally.
CODE_ENTITIES = {character: f"&#{ord(character)};" for character in "&<>\"'\\`*_[]|~#"}


def summary_prose(value: str) -> str:
    """Render untrusted text as inert single-line Markdown."""

    text = " ".join(value.split())
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#"):
        text = text.replace(character, f"\\{character}")
    return text


def summary_code(value: str) -> str:
    """Render untrusted text as inline code, also safe inside a table cell."""

    text = " ".join(value.split())
    return "<code>" + "".join(CODE_ENTITIES.get(c, c) for c in text) + "</code>"


def summary_diff(excerpt: str) -> list[str]:
    """Render an untrusted diff excerpt as fenced lines.

    The fence is longer than any backtick run in the excerpt, so no excerpt
    line can close it early.
    """

    longest = max((len(run) for run in re.findall(r"`+", excerpt)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}diff", *excerpt.splitlines(), fence]


def print_log_json(*, prefix: str, value: Any) -> None:
    """Print JSON to the job log so no line can run a workflow command.

    The runner parses `::command::` at the start of a line and the legacy
    `##[command]` anywhere in a line, so both are escaped.
    """

    for line in json.dumps(value, indent=2, sort_keys=True).splitlines():
        safe_line = line.replace("::", r"\u003a\u003a").replace("##[", r"\u0023\u0023[")
        print(f"{prefix} | {safe_line}", flush=True)
