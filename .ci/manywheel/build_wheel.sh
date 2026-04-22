#!/usr/bin/env bash
# Build a PyTorch wheel inside a manylinux container.
#
# Usage: build_wheel.sh <output_dir>
#
# Expects all build env vars (USE_CUDA, TORCH_CUDA_ARCH_LIST, etc.)
# to be set by the caller (GitHub Actions workflow env).

set -ex

OUTPUT_DIR="$1"

ARCH=$(uname -m)

# MKL cmake paths (x86_64 only)
if [[ -d /opt/intel/include ]]; then
    export CMAKE_INCLUDE_PATH="/opt/intel/include"
    export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib"
fi

# aarch64 CUDA: use NVPL for BLAS/LAPACK and wire ACL into oneDNN.
# Without BLAS=NVPL, CMake falls back to searching for MKL (x86-only) and
# oneDNN is built without ARM Compute Library acceleration. Exported here
# so they land in the same shell as the cmake invocation below.
if [[ "$ARCH" == "aarch64" && "${GPU_ARCH_TYPE:-}" == "cuda-aarch64" ]]; then
    if [[ ! -f /usr/local/lib/libnvpl_blas_lp64_gomp.so.0 ]]; then
        echo "ERROR: NVPL BLAS not found at /usr/local/lib/libnvpl_blas_lp64_gomp.so.0"
        exit 1
    fi
    if [[ ! -d /acl ]]; then
        echo "ERROR: ARM Compute Library not found at /acl"
        exit 1
    fi
    echo "Using NVPL BLAS/LAPACK and ACL for MKLDNN on CUDA aarch64"
    export BLAS=NVPL
    export USE_MKLDNN=1
    export USE_MKLDNN_ACL=1
    export ACL_ROOT_DIR=/acl
fi

set -u
pip install build
python -m build --wheel --no-isolation --outdir "$OUTPUT_DIR"
