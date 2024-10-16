#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2019

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
fi

pushd "$BUILDER_ROOT"

./windows/internal/smoke_test.bat

popd
