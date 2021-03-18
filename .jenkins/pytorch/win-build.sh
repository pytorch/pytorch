#!/bin/bash

# If you want to rebuild, run this with REBUILD=1
# If you want to build with CUDA, run this with USE_CUDA=1
# If you want to build without CUDA, run this with USE_CUDA=0

if [ ! -f setup.py ]; then
  echo "ERROR: Please run this build script from PyTorch root directory."
  exit 1
fi

# shellcheck disable=SC2034
COMPACT_JOB_NAME=pytorch-win-ws2019-cuda10-cudnn7-py3-build

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

export IMAGE_COMMIT_ID=$(git rev-parse HEAD)
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
export TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export PYTORCH_FINAL_PACKAGE_DIR="/c/w/build-results"
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
fi

# This directory is used only to hold "pytorch_env_restore.bat", called via "setup_pytorch_env.bat"
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p $CI_SCRIPTS_DIR

if [ -n "$(ls $CI_SCRIPTS_DIR/*)" ]; then
    rm $CI_SCRIPTS_DIR/*
fi

export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

set +ex
grep -E -R 'PyLong_(From|As)(Unsigned|)Long\(' --exclude=python_numbers.h torch/
PYLONG_API_CHECK=$?
if [[ $PYLONG_API_CHECK == 0 ]]; then
  echo "Usage of PyLong_{From,As}{Unsigned}Long API may lead to overflow errors on Windows"
  echo "because \`sizeof(long) == 4\` and \`sizeof(unsigned long) == 4\`."
  echo "Please include \"torch/csrc/python_numbers.h\" and use the correspoding APIs instead."
  echo "PyLong_FromLong -> THPUtils_packInt32 / THPUtils_packInt64"
  echo "PyLong_AsLong -> THPUtils_unpackInt (32-bit) / THPUtils_unpackLong (64-bit)"
  echo "PyLong_FromUnsignedLong -> THPUtils_packUInt32 / THPUtils_packUInt64"
  echo "PyLong_AsUnsignedLong -> THPUtils_unpackUInt32 / THPUtils_unpackUInt64"
  exit 1
fi
set -ex

$SCRIPT_HELPERS_DIR/build_pytorch.bat

assert_git_not_dirty

if [ ! -f ${TMP_DIR}/${IMAGE_COMMIT_TAG}.7z ] && [ ! ${BUILD_ENVIRONMENT} == "" ]; then
    exit 1
fi
echo "BUILD PASSED"
