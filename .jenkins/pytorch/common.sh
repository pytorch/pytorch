#!/bin/bash

# Common setup for all Jenkins scripts
# shellcheck source=./common_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"
set -ex

# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

# Figure out which Python to use for ROCm
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
  # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
  unset HIP_PLATFORM
  export PYTORCH_TEST_WITH_ROCM=1
  # temporary to locate some kernel issues on the CI nodes
  export HSAKMT_DEBUG_LEVEL=4
  # improve rccl performance for distributed tests
  export HSA_FORCE_FINE_GRAIN_PCIE=1
fi

# TODO: Renable libtorch testing for MacOS, see https://github.com/pytorch/pytorch/issues/62598
# shellcheck disable=SC2034
BUILD_TEST_LIBTORCH=0

# Use conda cmake in some CI build. Conda cmake will be newer than our supported
# min version (3.5 for xenial and 3.10 for bionic),
# so we only do it in four builds that we know should use conda.
# Linux bionic cannot find conda mkl with cmake 3.10, so we need a cmake from conda.
# Alternatively we could point cmake to the right place
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
if [[ "${TEST_CONFIG:-}" == *xla* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *centos* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *linux-bionic* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *linux-focal* ]]; then
  if ! which conda; then
    echo "Expected ${BUILD_ENVIRONMENT} to use conda, but 'which conda' returns empty"
    exit 1
  else
    conda install -q -y cmake
  fi
  if [[ "$BUILD_ENVIRONMENT" == *centos* ]]; then
    # cmake3 package will conflict with conda cmake
    sudo yum -y remove cmake3 || true
  fi
fi

retry () {
  "$@"  || (sleep 1 && "$@") || (sleep 2 && "$@")
}
