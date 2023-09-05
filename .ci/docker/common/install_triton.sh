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

if [ -z "${MAX_JOBS}" ]; then
    export MAX_JOBS=$(nproc)
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

if [ -n "${CONDA_CMAKE}" ]; then
  # TODO: This is to make sure that the same cmake and numpy version from install conda
  # script is used. Without this step, the newer cmake version (3.25.2) downloaded by
  # triton build step via pip will fail to detect conda MKL. Once that issue is fixed,
  # this can be removed.
  #
  # The correct numpy version also needs to be set here because conda claims that it
  # causes inconsistent environment.  Without this, conda will attempt to install the
  # latest numpy version, which fails ASAN tests with the following import error: Numba
  # needs NumPy 1.20 or less.
  conda_reinstall cmake="${CMAKE_VERSION}"
  conda_reinstall numpy="${NUMPY_VERSION}"
fi
