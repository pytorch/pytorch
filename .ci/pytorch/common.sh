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

# Bootstrap pip and setuptools into the per-Python-version conda env if they
# are missing. Miniforge >=26.3.x ships conda 26.3+, which no longer pulls
# pip/setuptools into envs created by `conda create python=X` from
# conda-forge. Docker images built against that Miniforge can end up with no
# pip/setuptools in /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/, which then
# breaks runtime invocations like `python -m pip install ...` and
# `python setup.py ...`. Fix that here at runtime so we don't need to rebuild
# the docker image.
if [[ -n "${ANACONDA_PYTHON_VERSION:-}" ]]; then
  CONDA_ENV_PREFIX="/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}"
  if [[ -x "${CONDA_ENV_PREFIX}/bin/python" ]] && \
     ! "${CONDA_ENV_PREFIX}/bin/python" -c 'import pip, setuptools' >/dev/null 2>&1; then
    echo "Bootstrapping pip and setuptools into ${CONDA_ENV_PREFIX}"
    conda install -n "py_${ANACONDA_PYTHON_VERSION}" -y -q pip setuptools
  fi
  unset CONDA_ENV_PREFIX
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
