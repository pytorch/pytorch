#!/bin/bash

set -ex

# Optionally install conda
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  BASE_URL="https://repo.anaconda.com/miniconda"

  MAJOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 1)
  MINOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 2)

  case "$MAJOR_PYTHON_VERSION" in
    2)
      CONDA_FILE="Miniconda2-latest-Linux-x86_64.sh"
    ;;
    3)
      CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
    ;;
    *)
      echo "Unsupported ANACONDA_PYTHON_VERSION: $ANACONDA_PYTHON_VERSION"
      exit 1
      ;;
  esac

  mkdir -p /opt/conda
  chown jenkins:jenkins /opt/conda

  source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  # NB: Manually invoke bash per https://github.com/conda/conda/issues/10431
  as_jenkins bash "${CONDA_FILE}" -b -f -p "/opt/conda"
  popd

  # NB: Don't do this, rely on the rpath to get it right
  #echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda-python.conf
  #ldconfig
  sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"

  # Ensure we run conda in a directory that jenkins has write access to
  pushd /opt/conda

  # Prevent conda from updating to 4.14.0, which causes docker build failures
  # See https://hud.pytorch.org/pytorch/pytorch/commit/754d7f05b6841e555cea5a4b2c505dd9e0baec1d
  # Uncomment the below when resolved to track the latest conda update
  # as_jenkins conda update -y -n base conda

  # Install correct Python version
  as_jenkins conda create -n py_$ANACONDA_PYTHON_VERSION -y python="$ANACONDA_PYTHON_VERSION"

  # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
  CONDA_COMMON_DEPS="astunparse pyyaml mkl=2021.4.0 mkl-include=2021.4.0 setuptools"
  if [ "$ANACONDA_PYTHON_VERSION" = "3.11" ]; then
    conda_install numpy=1.23.5 ${CONDA_COMMON_DEPS}
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.10" ]; then
    conda_install numpy=1.21.2 ${CONDA_COMMON_DEPS}
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.9" ]; then
    conda_install numpy=1.21.2 ${CONDA_COMMON_DEPS}
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.8" ]; then
    conda_install numpy=1.21.2 ${CONDA_COMMON_DEPS}
  else
    # Install `typing-extensions` for 3.7
    conda_install numpy=1.21.2 ${CONDA_COMMON_DEPS} typing-extensions
  fi

  # This is only supported in 3.8 upward
  if [ "$MINOR_PYTHON_VERSION" -gt "7" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    # and libpython-static for torch deploy
    conda_install llvmdev=8.0.0 "libpython-static=${ANACONDA_PYTHON_VERSION}"
  fi

  # Use conda cmake in some cases. Conda cmake will be newer than our supported
  # min version (3.5 for xenial and 3.10 for bionic), so we only do it in those
  # following builds that we know should use conda. Specifically, Ubuntu bionic
  # and focal cannot find conda mkl with stock cmake, so we need a cmake from conda
  if [ -n "${CONDA_CMAKE}" ]; then
    conda_install cmake
  fi

  # Magma package names are concatenation of CUDA major and minor ignoring revision
  # I.e. magma-cuda102 package corresponds to CUDA_VERSION=10.2 and CUDA_VERSION=10.2.89
  if [ -n "$CUDA_VERSION" ]; then
    conda_install magma-cuda$(TMP=${CUDA_VERSION/./};echo ${TMP%.*[0-9]}) -c pytorch
  fi

  # Install some other packages, including those needed for Python test reporting
  pip_install -r /opt/conda/requirements-ci.txt

  # Update scikit-learn to a python-3.8 compatible version
  if [[ $(python -c "import sys; print(int(sys.version_info >= (3, 8)))") == "1" ]]; then
    pip_install -U scikit-learn
  else
    # Pinned scikit-learn due to https://github.com/scikit-learn/scikit-learn/issues/14485 (affects gcc 5.5 only)
    pip_install scikit-learn==0.20.3
  fi

  popd
fi
