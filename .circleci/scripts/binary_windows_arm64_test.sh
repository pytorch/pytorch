#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export CUDA_VERSION="cpu"
export VC_YEAR=2022

pushd "$BUILDER_ROOT"

./windows/internal/arm64/smoke_test.bat

popd
