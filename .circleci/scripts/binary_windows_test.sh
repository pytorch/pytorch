#!/bin/bash
set -eux -o pipefail

source "/c/w/env"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2017

if [[ "$CUDA_VERSION" == "92" || "$CUDA_VERSION" == "100" ]]; then
  export VC_YEAR=2017
else
  export VC_YEAR=2019
fi

pushd "$BUILDER_ROOT"

./windows/internal/smoke_test.bat

popd
