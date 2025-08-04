
#!/usr/bin/env bash
set -euo pipefail

TARGET_FILE="requirements/test.in"
TMP_HEAD=$(mktemp)

PKGS=("torch" "torchvision" "torchaudio" "xformers" "mamba_ssm" "schemathesis")

echo "[INFO] Scanning and removing the following local packages from $TARGET_FILE:"
for pkg in "${PKGS[@]}"; do
  # Show matches for info
  MATCHES=$(grep -Ei "^${pkg}[[:space:]]*(@|==)" "$TARGET_FILE" || true)
  if [[ -n "$MATCHES" ]]; then
    echo "Removing: $MATCHES"
    # Remove lines like:
    # torch==...
    # torch == ...
    # torch @ ...
    # torch@ ...
    # torch>=...
    sed -i "/^${pkg}[[:space:]]*==/Id" "$TARGET_FILE"
    sed -i "/^${pkg}[[:space:]]*@/Id" "$TARGET_FILE"
    sed -i "/^${pkg}[[:space:]]*>=/Id" "$TARGET_FILE"
  fi
done

echo "[INFO] Done."

pip freeze | grep -iE '^(torch|torchvision|torchaudio)\s+@ file://' > "$TMP_HEAD"
echo "schemathesis==3.39.15" >> "$TMP_HEAD"


echo "" >> "$TMP_HEAD"
cat "$TARGET_FILE" >> "$TMP_HEAD"
mv "$TMP_HEAD" "$TARGET_FILE"

echo "[INFO] Successfully prepended local .whl references to $TARGET_FILE:"
head -n 10 "$TARGET_FILE"
