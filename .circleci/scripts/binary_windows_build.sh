#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache
export SCCACHE_IGNORE_SERVER_IO_ERROR=1
export VC_YEAR=2019

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
    export USE_SCCACHE=0
fi

echo "Free space on filesystem before build:"
df -h

pushd "$BUILDER_ROOT"
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
    ./windows/internal/build_conda.bat
elif [[ "$PACKAGE_TYPE" == 'wheel' || "$PACKAGE_TYPE" == 'libtorch' ]]; then
    export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"
    ./windows/internal/build_wheels.bat
fi

echo "Free space on filesystem after build:"
df -h
