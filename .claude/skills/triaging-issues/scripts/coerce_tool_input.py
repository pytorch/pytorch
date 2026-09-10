#!/usr/bin/env python3
"""PreToolUse hook that repairs stringified MCP arguments.

With deferred MCP tool schemas the model guesses argument types: it sends
issue_number as "195701" and labels as the JSON text '["module: cuda"]'.
The pinned github-mcp-server asserts float64 / array and rejects the call,
so a run can end with every GitHub call failed. Coerce and pass the call
through.

Exit codes:
  0 - Allow the tool call (with input rewrite via stdout JSON when needed)
"""

import json
import sys


INT_FIELDS = ("issue_number", "page", "perPage")
LIST_FIELDS = ("labels", "assignees")


def coerced(tool_input: dict) -> dict | None:
    updated = dict(tool_input)
    for field in INT_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value.strip().lstrip("#").isdigit():
            updated[field] = int(value.strip().lstrip("#"))
    for field in LIST_FIELDS:
        value = tool_input.get(field)
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            updated[field] = parsed
    return None if updated == tool_input else updated


def main() -> None:
    data = json.load(sys.stdin)
    updated = coerced(data.get("tool_input", {}))
    if updated is None:
        return
    changed = sorted(k for k in updated if updated[k] != data["tool_input"].get(k))
    print(f"Coerced stringified fields: {changed}", file=sys.stderr)
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
