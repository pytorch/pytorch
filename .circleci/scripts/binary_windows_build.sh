#!/bin/bash
set -eux -o pipefail

source "/c/w/env"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export VC_YEAR=2017
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache-windows
export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"

set +x
export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}
set -x

if [[ "$CIRCLECI" == 'true' && -d "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019" ]]; then
  rm -rf "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019"
fi

echo "Free space on filesystem before build:"
df -h

pushd "$BUILDER_ROOT"
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  ./windows/internal/build_conda.bat
elif [[ "$PACKAGE_TYPE" == 'wheel' || "$PACKAGE_TYPE" == 'libtorch' ]]; then
  ./windows/internal/build_wheels.bat
fi

echo "Free space on filesystem after build:"
df -h
