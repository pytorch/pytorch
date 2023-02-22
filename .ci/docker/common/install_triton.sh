#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit triton)

apt update
apt-get install -y gpg-agent

if [ -n "${CONDA_CMAKE}" ]; then
  # TODO: This is to make sure that the same cmake version from install_conda.sh
  # is used. Without this step, triton build will download the newer cmake version
  # (3.25.2) which fails to detect conda MKL. Once that issue is fixed, this can
  # be removed
  conda_install cmake
fi

if [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  CXX=g++-9 pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
elif [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  CXX=g++-9 pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
else
  pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
fi
