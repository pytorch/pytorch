#!/bin/bash

# If you want to rebuild, run this with REBUILD=1
# If you want to build with CUDA, run this with USE_CUDA=1
# If you want to build without CUDA, run this with USE_CUDA=0

if [ ! -f setup.py ]; then
  echo "ERROR: Please run this build script from PyTorch root directory."
  exit 1
fi

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"
# shellcheck source=./common-build.sh
source "$SCRIPT_PARENT_DIR/common-build.sh"

export TMP_DIR="${PWD}/build/win_tmp"
TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export TMP_DIR_WIN
export PYTORCH_FINAL_PACKAGE_DIR=${PYTORCH_FINAL_PACKAGE_DIR:-/c/w/build-results}
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
fi

export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-arm64-test-helpers

"$SCRIPT_HELPERS_DIR"/build_pytorch.bat

assert_git_not_dirty

echo "BUILD PASSED"
