#!/usr/bin/env bash
# Minimal include shim for cross-compiling torch_python against Debian arm64 Python.
# pyconfig.h includes <aarch64-linux-gnu/python3.11/pyconfig.h> without pulling full Debian usr/include.
set -euo pipefail
ROOTFS="${1:?usage: setup_python_cross_include.sh ROOTFS OUT_DIR}"
OUT="${2:?usage: setup_python_cross_include.sh ROOTFS OUT_DIR}"
mkdir -p "$OUT/aarch64-linux-gnu/python3.13"
cp "$ROOTFS/usr/include/aarch64-linux-gnu/python3.13/pyconfig.h" \
  "$OUT/aarch64-linux-gnu/python3.13/pyconfig.h"
