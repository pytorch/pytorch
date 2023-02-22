#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit triton)

apt update
if [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  export CXX=g++-9
  pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
elif [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  export CXX=g++-9
  pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
else
  pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
fi
