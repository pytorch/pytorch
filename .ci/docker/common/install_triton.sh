#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

get_conda_version() {
  as_jenkins conda list -n py_$ANACONDA_PYTHON_VERSION | grep -w $* | head -n 1 | awk '{print $2}'
}

conda_reinstall() {
  as_jenkins conda install -q -n py_$ANACONDA_PYTHON_VERSION -y --force-reinstall $*
}

if [ -n "${XPU_VERSION}" ]; then
  TRITON_REPO="https://github.com/intel/intel-xpu-backend-for-triton"
  TRITON_TEXT_FILE="triton-xpu"
elif [ -n "${TRITON_CPU}" ]; then
  TRITON_REPO="https://github.com/triton-lang/triton-cpu"
  TRITON_TEXT_FILE="triton-cpu"
else
  TRITON_REPO="https://github.com/triton-lang/triton"
  TRITON_TEXT_FILE="triton"
fi

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit ${TRITON_TEXT_FILE})

if [ -n "${UBUNTU_VERSION}" ];then
    apt update
    apt-get install -y gpg-agent
fi

if [ -n "${CONDA_CMAKE}" ]; then
  # Keep the current cmake and numpy version here, so we can reinstall them later
  CMAKE_VERSION=$(get_conda_version cmake)
  NUMPY_VERSION=$(get_conda_version numpy)
fi

if [ -z "${MAX_JOBS}" ]; then
    export MAX_JOBS=$(nproc)
fi

# Git checkout triton
mkdir /var/lib/jenkins/triton
chown -R jenkins /var/lib/jenkins/triton
chgrp -R jenkins /var/lib/jenkins/triton
pushd /var/lib/jenkins/

as_jenkins git clone --recursive ${TRITON_REPO} triton
cd triton
as_jenkins git checkout ${TRITON_PINNED_COMMIT}
as_jenkins git submodule update --init --recursive
cd python

# TODO: remove patch setup.py once we have a proper fix for https://github.com/triton-lang/triton/issues/4527
as_jenkins sed -i -e 's/https:\/\/tritonlang.blob.core.windows.net\/llvm-builds/https:\/\/oaitriton.blob.core.windows.net\/public\/llvm-builds/g' setup.py

if [ -n "${UBUNTU_VERSION}" ] && [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  CXX=g++-9 pip_install -e .
elif [ -n "${UBUNTU_VERSION}" ] && [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  CXX=g++-9 pip_install -e .
else
  pip_install -e .
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
  # Note that we install numpy with pip as conda might not have the version we want
  pip_install --force-reinstall numpy=="${NUMPY_VERSION}"
fi
