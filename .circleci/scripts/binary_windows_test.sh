#!/bin/bash
set -eux -o pipefail

source "/c/w/env"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"

pushd "$BUILDER_ROOT"

./windows/internal/smoke_test.bat

popd
