#!/bin/bash

TARGET_FILE="test.in"
TMP_HEAD=$(mktemp)

for pkg in torch torchvision torchaudio xformers flashinfer-python; do
    pip freeze | grep -E "^${pkg}[[:space:]]+@ file://" >> "$TMP_HEAD"
done

echo "" >> "$TMP_HEAD"
cat "$TARGET_FILE" >> "$TMP_HEAD"

mv "$TMP_HEAD" "$TARGET_FILE"

echo "[INFO] Local wheel requirements prepended to $TARGET_FILE"
