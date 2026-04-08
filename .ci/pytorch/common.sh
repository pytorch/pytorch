#!/bin/bash

# Common setup for all Jenkins scripts
# shellcheck source=./common_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"
set -ex -o pipefail

# Source ROCm environment variables (paths may vary between tarball/wheel installs)
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]] && [[ -f /etc/rocm_env.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/rocm_env.sh
fi

# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

# Select compiler based on build environment name. Images that have both
# GCC and Clang installed default cc/c++ to Clang (via install_clang.sh),
# so we need to override when a gcc build is requested.
if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
  export CC=clang
  export CXX=clang++
elif [[ "${BUILD_ENVIRONMENT}" == *gcc* ]]; then
  export CC=gcc
  export CXX=g++
  sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100
  sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100
fi

# Figure out which Python to use for ROCm
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
  # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
  unset HIP_PLATFORM
  export PYTORCH_TEST_WITH_ROCM=1
fi

# TODO: Reenable libtorch testing for MacOS, see https://github.com/pytorch/pytorch/issues/62598
# shellcheck disable=SC2034
BUILD_TEST_LIBTORCH=0
