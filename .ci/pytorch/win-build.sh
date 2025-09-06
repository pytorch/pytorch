#!/bin/bash

# If you want to rebuild, run this with REBUILD=1
# If you want to build with CUDA, run this with USE_CUDA=1
# If you want to build without CUDA, run this with USE_CUDA=0
# If you want to build with ROCm, run this with BUILD_ENVIRONMENT=*-rocm-*

if [ ! -f setup.py ]; then
  echo "ERROR: Please run this build script from PyTorch root directory."
  exit 1
fi

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"
# shellcheck source=./common-build.sh
source "$SCRIPT_PARENT_DIR/common-build.sh"

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # Note: this assumes that the rocm python packages are already installed.

  # shellcheck source=./rocm_sdk-build.sh
  source "$SCRIPT_PARENT_DIR/rocm_sdk-build.sh"

  # Prepare sources: hipify then write bootstrap script.
  python tools/amd_build/build_amd.py
  python tools/amd_build/write_rocm_init.py
fi

export TMP_DIR="${PWD}/build/win_tmp"
TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export TMP_DIR_WIN
export PYTORCH_FINAL_PACKAGE_DIR=${PYTORCH_FINAL_PACKAGE_DIR:-/c/w/build-results}
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
fi

export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

set +ex
grep -E -R 'PyLong_(From|As)(Unsigned|)Long\(' --exclude=python_numbers.h  --exclude=pythoncapi_compat.h --exclude=eval_frame.c torch/
PYLONG_API_CHECK=$?
if [[ $PYLONG_API_CHECK == 0 ]]; then
  echo "Usage of PyLong_{From,As}{Unsigned}Long API may lead to overflow errors on Windows"
  echo "because \`sizeof(long) == 4\` and \`sizeof(unsigned long) == 4\`."
  echo "Please include \"torch/csrc/utils/python_numbers.h\" and use the corresponding APIs instead."
  echo "PyLong_FromLong -> THPUtils_packInt32 / THPUtils_packInt64"
  echo "PyLong_AsLong -> THPUtils_unpackInt (32-bit) / THPUtils_unpackLong (64-bit)"
  echo "PyLong_FromUnsignedLong -> THPUtils_packUInt32 / THPUtils_packUInt64"
  echo "PyLong_AsUnsignedLong -> THPUtils_unpackUInt32 / THPUtils_unpackUInt64"
  exit 1
fi
set -ex -o pipefail

"$SCRIPT_HELPERS_DIR"/build_pytorch.bat

assert_git_not_dirty

echo "BUILD PASSED"
