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
echo "build_wheel.sh: ARCH=$ARCH GPU_ARCH_TYPE=${GPU_ARCH_TYPE:-unset} DESIRED_CUDA=${DESIRED_CUDA:-unset}"

if [[ "$ARCH" == "x86_64" ]]; then
    # MKL is installed at /opt/intel by .ci/docker/common/install_mkl.sh
    if [[ -d /opt/intel/include ]]; then
        export CMAKE_INCLUDE_PATH="/opt/intel/include"
        export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib"
    fi
elif [[ "$ARCH" == "aarch64" ]]; then
    # Use NVPL (CUDA) or OpenBLAS (CPU) for BLAS/LAPACK and wire ACL into
    # oneDNN. Without an explicit BLAS choice, CMake falls back to searching
    # for MKL (x86-only) and oneDNN is built without ACL acceleration.
    if [[ ! -d /acl ]]; then
        echo "ERROR: ARM Compute Library not found at /acl"
        exit 1
    fi
    export USE_MKLDNN=1
    export USE_MKLDNN_ACL=1
    export ACL_ROOT_DIR=/acl

    case "${GPU_ARCH_TYPE:-}" in
        cuda-aarch64)
            if [[ ! -f /usr/local/lib/libnvpl_blas_lp64_gomp.so.0 ]]; then
                echo "ERROR: NVPL BLAS not found at /usr/local/lib/libnvpl_blas_lp64_gomp.so.0"
                exit 1
            fi
            echo "Using NVPL BLAS/LAPACK and ACL for MKLDNN on CUDA aarch64"
            export BLAS=NVPL
            ;;
        cpu-aarch64|cpu)
            if [[ ! -f /opt/OpenBLAS/lib/libopenblas.so.0 ]]; then
                echo "ERROR: OpenBLAS not found at /opt/OpenBLAS/lib/libopenblas.so.0"
                exit 1
            fi
            echo "Using OpenBLAS and ACL for MKLDNN on CPU aarch64"
            export BLAS=OpenBLAS
            export OpenBLAS_HOME=/opt/OpenBLAS
            ;;
    esac
fi

set -u
pip install build
python -m build --wheel --no-isolation --outdir "$OUTPUT_DIR"
