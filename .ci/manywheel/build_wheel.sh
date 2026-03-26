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

# Source vendor environment scripts for XPU/ROCm builds.
# These don't persist from the GPU environment setup step.
if [[ "${GPU_ARCH_TYPE:-}" == "xpu" ]]; then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null || true
    source /opt/intel/oneapi/pti/latest/env/vars.sh 2>/dev/null || true
    source /opt/intel/oneapi/umf/latest/env/vars.sh 2>/dev/null || true
    source /opt/intel/oneapi/ccl/latest/env/vars.sh 2>/dev/null || true
    source /opt/intel/oneapi/mpi/latest/env/vars.sh 2>/dev/null || true
fi
if [[ "${GPU_ARCH_TYPE:-}" == "rocm" ]]; then
    [[ -f /etc/rocm_env.sh ]] && source /etc/rocm_env.sh
fi

set -u
pip install build
python -m build --wheel --no-isolation --outdir "$OUTPUT_DIR"
