#!/usr/bin/env bash
# Build a PyTorch wheel inside a manylinux container.
#
# Usage: build_wheel.sh <output_dir>
#
# Expects all build env vars (USE_CUDA, TORCH_CUDA_ARCH_LIST, etc.)
# to be set by the caller (GitHub Actions workflow env).

set -ex

OUTPUT_DIR="$1"

# MKL cmake paths (x86_64 only)
if [[ -d /opt/intel/include ]]; then
    export CMAKE_INCLUDE_PATH="/opt/intel/include"
    export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib"
fi

set -u
pip install build
python -m build --wheel --no-isolation --outdir "$OUTPUT_DIR"
