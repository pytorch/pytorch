#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2019

pushd "$BUILDER_ROOT"
if [[ ${BUILD_SHARED_LIBS:-true} == "false" ]]; then
    ./windows/internal/static_lib_test.bat
else
    ./windows/internal/smoke_test.bat
fi

popd
