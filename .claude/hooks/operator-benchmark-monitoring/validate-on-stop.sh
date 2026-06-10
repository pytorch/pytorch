#!/usr/bin/env bash
# Stop hook to validate that output file exists and is valid

set -euo pipefail

OUTPUT_FILE="/tmp/benchmark-regression-actions.json"

if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "ERROR: Output file not found at $OUTPUT_FILE" >&2
    echo "The monitoring must produce a benchmark-regression-actions.json file" >&2
    exit 1
fi

# Validate JSON
if ! jq empty "$OUTPUT_FILE" 2>/dev/null; then
    echo "ERROR: Invalid JSON in output file" >&2
    exit 1
fi

# Validate actions array exists
if ! jq -e '.actions | type == "array"' "$OUTPUT_FILE" >/dev/null 2>&1; then
    echo "ERROR: Missing 'actions' array" >&2
    exit 1
fi

echo "✓ Final validation passed" >&2
exit 0
