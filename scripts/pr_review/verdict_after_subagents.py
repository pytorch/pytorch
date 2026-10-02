#!/usr/bin/env python3
"""Refuse a findings file written before the reviewer's sub-agents reported.

The review prompt asks for an early findings file so the validator's feedback
arrives while there is time to act on it. With sub-agents that early file is a
draft: if the reviewer ends its turn without rewriting it after their reports,
the draft is what gets published. On CI that published "Draft; review in
progress." marked ready.

Reads the main session transcript (sub-agent turns live in separate sidechain
files) and compares the position of the reviewer's last Write to the findings
file with the last sub-agent result. Only a Write whose result came back
without error counts: a failed rewrite leaves the draft in place. Exit 0 when
the write is the later of the two, or no sub-agent ran; exit 1 when a sub-agent
reported after it, or nothing was ever written after one ran; exit 2 when the
transcript cannot be read (the Stop hook logs that and does not block).

Sub-agents must run in the foreground (the workflow disables background tasks),
so an Agent tool_result is the sub-agent's report. The findings path is matched
exactly because restrict-write.sh only allows that exact spelling.

Usage: verdict_after_subagents.py <transcript.jsonl> <findings-path>
"""

from __future__ import annotations

import json
import sys


def last_positions(lines, findings: str) -> tuple[int, int]:
    """(index of the last successful findings Write result, of the last sub-agent
    result); -1 if none."""
    agent_ids: set[str] = set()
    write_ids: set[str] = set()
    last_write = last_agent = -1
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("isSidechain"):
            continue
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_use":
                name = item.get("name")
                args = item.get("input") if isinstance(item.get("input"), dict) else {}
                if name in ("Agent", "Task"):
                    agent_ids.add(item.get("id"))
                elif name == "Write" and args.get("file_path") == findings:
                    write_ids.add(item.get("id"))
            elif item.get("type") == "tool_result":
                tool_id = item.get("tool_use_id")
                if tool_id in agent_ids:
                    last_agent = i
                elif tool_id in write_ids and not item.get("is_error"):
                    last_write = i
    return last_write, last_agent


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        with open(argv[1], encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"cannot read transcript: {exc}", file=sys.stderr)
        return 2
    last_write, last_agent = last_positions(lines, argv[2])
    if last_agent > last_write:
        print(
            "a sub-agent reported after your last write of the findings file",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
