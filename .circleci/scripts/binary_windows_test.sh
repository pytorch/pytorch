#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="${GPU_ARCH_VERSION:-cpu}"
export VC_YEAR=2019

pushd "$BUILDER_ROOT"

./windows/internal/smoke_test.bat

popd
