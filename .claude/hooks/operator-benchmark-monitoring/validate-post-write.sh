#!/usr/bin/env bash
# Post-hook to validate JSON structure after write

set -euo pipefail

OUTPUT_FILE="/tmp/benchmark-regression-actions.json"

if [[ ! -f "$OUTPUT_FILE" ]]; then
    # File doesn't exist yet, allow
    exit 0
fi

# Validate JSON structure
if ! jq empty "$OUTPUT_FILE" 2>/dev/null; then
    echo "ERROR: Invalid JSON in output file" >&2
    exit 1
fi

# Check for actions array
if ! jq -e '.actions | type == "array"' "$OUTPUT_FILE" >/dev/null 2>&1; then
    echo "ERROR: Missing 'actions' array" >&2
    exit 1
fi

# Validate each action has required fields
COUNT=$(jq '.actions | length' "$OUTPUT_FILE")
for ((i=0; i<COUNT; i++)); do
    TYPE=$(jq -r ".actions[$i].type // empty" "$OUTPUT_FILE")

    if [[ -z "$TYPE" ]]; then
        echo "ERROR: Action $i missing 'type' field" >&2
        exit 1
    fi

    if [[ "$TYPE" != "create" && "$TYPE" != "update" && "$TYPE" != "close" && "$TYPE" != "noop" ]]; then
        echo "ERROR: Invalid action type: $TYPE" >&2
        exit 1
    fi

    # Validate repo field
    REPO=$(jq -r ".actions[$i].repo // empty" "$OUTPUT_FILE")
    if [[ "$REPO" != "pytorch/pytorch" ]]; then
        echo "ERROR: Action $i: repo must be 'pytorch/pytorch', got '$REPO'" >&2
        exit 1
    fi
done

echo "✓ Output validated successfully" >&2
exit 0
