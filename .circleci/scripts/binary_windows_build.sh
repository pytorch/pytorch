#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache
export SCCACHE_IGNORE_SERVER_IO_ERROR=1
export VC_YEAR=2022

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export USE_SCCACHE=0
    export XPU_VERSION=2025.0
    export XPU_ENABLE_KINETO=1
fi

echo "Free space on filesystem before build:"
df -h

pushd "$PYTORCH_ROOT/.ci/pytorch/"
export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"

if [[ os == 'windows' ]]; then
    ./windows/internal/build_wheels.bat
elif [[ os == 'windows-arm64' ]]; then{
    if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
        ./windows/arm64/build_libtorch.bat
    elif [[ "$PACKAGE_TYPE" == 'wheel' ]]; then
        ./windows/arm64/build_pytorch.bat
    fi
fi

echo "Free space on filesystem after build:"
df -h
