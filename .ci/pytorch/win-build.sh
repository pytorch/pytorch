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

IMAGE_COMMIT_ID=$(git rev-parse HEAD)
export IMAGE_COMMIT_ID
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export TMP_DIR_WIN
export PYTORCH_FINAL_PACKAGE_DIR=${PYTORCH_FINAL_PACKAGE_DIR:-/c/w/build-results}
if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
fi

# This directory is used only to hold "pytorch_env_restore.bat", called via "setup_pytorch_env.py"
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p "$CI_SCRIPTS_DIR"

if [ -n "$(ls "$CI_SCRIPTS_DIR"/*)" ]; then
    rm "$CI_SCRIPTS_DIR"/*
fi

export SCRIPT_HELPERS_DIR=${SCRIPT_PARENT_DIR}/win-test-helpers

# These variables are set for the python build
if [ $DEBUG == "1" ]
then
  export BUILD_TYPE="debug"
else
  export BUILD_TYPE="release"
fi

if [ $BUILD_ENVIRONMENT == "1" ]
then
  export CONDA_PARENT_DIR=${(dirname $0)}"\win-test-helpers"
else
  export CONDA_PARENT_DIR="C:\Jenkins"
fi

export PATH="C:\Program^ Files\CMake\bin;C:\Program^ Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program^ Files\Git\cmd;C:\Program^ Files\Amazon\AWSCLI;C:\Program^ Files\Amazon\AWSCLI\bin;"${PATH}

export INSTALLER_DIR=${SCRIPT_HELPERS_DIR}"\installation-helpers"

export CMAKE_INCLUDE_PATH=${TMP_DIR_WIN}"\mkl\include"

export LIB=${TMP_DIR_WIN}"\mkl\lib;"$LIB

export INSTALL_FRESH_CONDA="1"

export PATH=${CONDA_PARENT_DIR}"\Miniconda3\Library\bin;"${CONDA_PARENT_DIR}"\Miniconda3;"${CONDA_PARENT_DIR}"\Miniconda3\Scripts;"${PATH}

export DISTUTILS_USE_SDK="1"

export PATH=${TMP_DIR_WIN}"\bin;"${PATH}

if [ $TORCH_CUDA_ARCH_LIST == "" ]
then
  export TORCH_CUDA_ARCH_LIST="5.2"
fi

export SCCACHE_IDLE_TIMEOUT="0"

export SCCACHE_IGNORE_SERVER_IO_ERROR="1"

export CC="sccache-cl"

export CXX="sccache-cl"

export CMAKE_GENERATOR="Ninja"

set +ex
grep -E -R 'PyLong_(From|As)(Unsigned|)Long\(' --exclude=python_numbers.h --exclude=eval_frame.c torch/
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

python ${SCRIPT_HELPERS_DIR}"/build_pytorch.py"

env

assert_git_not_dirty

if [ ! -f "${TMP_DIR}"/"${IMAGE_COMMIT_TAG}".7z ] && [ ! "${BUILD_ENVIRONMENT}" == "" ]; then
    exit 1
fi
echo "BUILD PASSED"
