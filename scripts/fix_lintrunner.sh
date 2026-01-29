#!/usr/bin/env bash
# Fix lintrunner setup: install uv if missing, then run lintrunner.
# Run this from the pytorch repo root in your terminal (not from Cursor's sandbox).
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Install uv if not found
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  else
    echo "Curlish install failed, trying: pip install uv"
    pip install uv
  fi
  if ! command -v uv &>/dev/null; then
    echo "uv still not found. Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "Then re-run this script."
    exit 1
  fi
fi

echo "Using uv: $(uv --version)"

# 2. Run lintrunner from repo root
echo "Running lintrunner -a on test/test_binary_ufuncs.py..."
lintrunner -a test/test_binary_ufuncs.py
