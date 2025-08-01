#!/bin/bash
set -euo pipefail

TARGET_FILE="requirements/test.in"
TMP_HEAD=$(mktemp)

pip freeze | grep -iE '^(torch|torchvision|torchaudio|xformers|flashinfer-python)\s+@ file://' > "$TMP_HEAD"

echo "" >> "$TMP_HEAD"
cat "$TARGET_FILE" >> "$TMP_HEAD"
mv "$TMP_HEAD" "$TARGET_FILE"

echo "[INFO] Successfully prepended local .whl references to $TARGET_FILE:"
head -n 10 "$TARGET_FILE"
