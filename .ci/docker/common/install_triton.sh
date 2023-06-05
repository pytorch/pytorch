#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

get_conda_version() {
  as_jenkins conda list -n py_$ANACONDA_PYTHON_VERSION | grep -w $* | head -n 1 | awk '{print $2}'
}

conda_reinstall() {
  as_jenkins conda install -q -n py_$ANACONDA_PYTHON_VERSION -y --force-reinstall $*
}

if [ -n "${ROCM_VERSION}" ]; then
  TRITON_REPO="https://github.com/ROCmSoftwarePlatform/triton"
  TRITON_TEXT_FILE="triton-rocm"
else
  TRITON_REPO="https://github.com/openai/triton"
  TRITON_TEXT_FILE="triton"
fi

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit ${TRITON_TEXT_FILE})

apt update
apt-get install -y gpg-agent

if [ -n "${CONDA_CMAKE}" ]; then
  # Keep the current cmake and numpy version here, so we can reinstall them later
  CMAKE_VERSION=$(get_conda_version cmake)
  NUMPY_VERSION=$(get_conda_version numpy)
fi

if [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  CXX=g++-9 pip_install "git+${TRITON_REPO}@${TRITON_PINNED_COMMIT}#subdirectory=python"
elif [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  CXX=g++-9 pip_install "git+${TRITON_REPO}@${TRITON_PINNED_COMMIT}#subdirectory=python"
else
  pip_install "git+${TRITON_REPO}@${TRITON_PINNED_COMMIT}#subdirectory=python"
fi