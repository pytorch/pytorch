#!/bin/bash

set -euo pipefail

TARGET_FILE="test.in"
TMP_HEAD=$(mktemp)

# Remove existing pinned torch/torchaudio/vision lines from target .in file
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' '/^torch==/d' "$TARGET_FILE"
  sed -i '' '/^torchaudio==/d' "$TARGET_FILE"
  sed -i '' '/^torchvision==/d' "$TARGET_FILE"
else
  sed -i '/^torch==/d' "$TARGET_FILE"
  sed -i '/^torchaudio==/d' "$TARGET_FILE"
  sed -i '/^torchvision==/d' "$TARGET_FILE"
fi

# Prepend local wheel to the test.in
for pkg in torch torchvision torchaudio xformers flashinfer-python; do
  pip freeze | grep -E "^${pkg}[[:space:]]+@ file://" >> "$TMP_HEAD"
done

echo "" >> "$TMP_HEAD"
cat "$TARGET_FILE" >> "$TMP_HEAD"

mv "$TMP_HEAD" "$TARGET_FILE"

echo "[INFO] Local wheel requirements prepended to $TARGET_FILE"
