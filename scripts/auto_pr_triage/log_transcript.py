"""Print the worker's Claude Code transcript to the job log.

The Claude action hides the conversation, and it writes no execution log when
the run fails (for example, after running out of structured-output retries).
Claude Code saves its own transcript as it runs, so this prints every user and
assistant message from it, with workflow commands neutralized. The first
message is the prepared prompt, which is omitted because it is large and
already recorded in llm_input.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from step_summary import print_log_json


def transcript_messages(path: Path) -> list[dict[str, Any]]:
    """Return the role and content of each conversation message in a transcript."""

    messages = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("type") in ("user", "assistant"):
            message = record.get("message", {})
            messages.append({key: message.get(key) for key in ("role", "content")})
    if messages and isinstance(messages[0]["content"], str):
        length = len(messages[0]["content"])
        messages[0]["content"] = f"<prepared prompt omitted: {length} characters>"
    return messages


def main() -> int:
    """Print each transcript under the Claude Code projects directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.projects_dir.glob("*/*.jsonl"))
    if not paths:
        print("No Claude transcript was written.")
    for path in paths:
        print_log_json(prefix="Claude transcript", value=transcript_messages(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
