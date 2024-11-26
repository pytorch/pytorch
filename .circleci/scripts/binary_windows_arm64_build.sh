#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

echo "Free space on filesystem before build:"
df -h

export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"

pushd "$BUILDER_ROOT"
if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
    ./windows/arm64/build_libtorch.bat
elif [[ "$PACKAGE_TYPE" == 'wheel' ]]; then
    ./windows/arm64/build_pytorch.bat
fi

echo "Free space on filesystem after build:"
df -h
