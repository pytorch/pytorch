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

# Bootstrap pip/setuptools and the rest of requirements-ci.txt into the
# per-Python-version conda env if they are missing. Miniforge >=26.3.x ships
# conda 26.3+, which no longer pulls pip/setuptools into envs created by
# `conda create python=X` from conda-forge. When that happens the
# image-build-time `pip install -r /opt/conda/requirements-ci.txt` in
# install_conda.sh runs in the base env (because there is no pip in
# py_$VER/bin yet, so `conda run` falls through to base), so all the pinned
# CI deps (`build`, `ninja`, `hypothesis`, ...) end up in the base env, not
# in /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/. Subsequent runtime calls
# like `python -m pip ...`, `python setup.py ...`, and
# `python -m build --wheel ...` then break. Fix that here at runtime so we
# don't need to rebuild the docker image.
if [[ -n "${ANACONDA_PYTHON_VERSION:-}" ]]; then
  CONDA_ENV_PREFIX="/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}"
  CONDA_ENV_PYTHON="${CONDA_ENV_PREFIX}/bin/python"
  if [[ -x "${CONDA_ENV_PYTHON}" ]]; then
    if ! "${CONDA_ENV_PYTHON}" -c 'import pip, setuptools' >/dev/null 2>&1; then
      echo "Bootstrapping pip and setuptools into ${CONDA_ENV_PREFIX}"
      conda install -n "py_${ANACONDA_PYTHON_VERSION}" -y -q pip setuptools
    fi
    # Use `build` as the canary for "did requirements-ci.txt land in this
    # env?" — it's pinned in that file but is not a default conda dep. The
    # Dockerfile rm's /opt/conda/requirements-ci.txt right after
    # install_conda.sh runs, so at runtime read the requirements file from
    # the checked-out source tree (.ci/docker/requirements-ci.txt) and fall
    # back to the in-image copy for older images that may still ship it.
    REQ_CI_TXT=""
    for candidate in \
      "$(dirname "${BASH_SOURCE[0]}")/../docker/requirements-ci.txt" \
      /opt/conda/requirements-ci.txt; do
      if [[ -f "$candidate" ]]; then
        REQ_CI_TXT="$candidate"
        break
      fi
    done
    if [[ -n "$REQ_CI_TXT" ]] && \
       ! "${CONDA_ENV_PYTHON}" -c 'import build' >/dev/null 2>&1; then
      echo "Re-installing $REQ_CI_TXT into ${CONDA_ENV_PREFIX}"
      "${CONDA_ENV_PYTHON}" -m pip install -r "$REQ_CI_TXT"
    fi
    unset REQ_CI_TXT
  fi
  unset CONDA_ENV_PREFIX CONDA_ENV_PYTHON
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
