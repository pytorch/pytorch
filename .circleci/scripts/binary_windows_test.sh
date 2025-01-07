#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2019

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
    export XPU_VERSION=2025.0
fi

pushd "$PYTORCH_ROOT/.ci/pytorch/"
./windows/internal/smoke_test.bat

popd
