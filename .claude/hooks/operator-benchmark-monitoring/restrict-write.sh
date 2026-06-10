#!/usr/bin/env bash
# Pre-hook to restrict Write tool to only /tmp/benchmark-regression-actions.json

set -euo pipefail

# Read the hook payload and get the file path from the Write tool call
input=$(cat)
FILE_PATH=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || echo "")

ALLOWED_FILE="/tmp/benchmark-regression-actions.json"

if [[ "$FILE_PATH" != "$ALLOWED_FILE" ]]; then
    echo "ERROR: Write tool is restricted to $ALLOWED_FILE only" >&2
    echo "Attempted to write to: $FILE_PATH" >&2
    exit 1
fi

# Allow the write
exit 0
