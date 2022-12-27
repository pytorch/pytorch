#!/bin/bash

set -ex

# Optionally install conda
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  BASE_URL="https://repo.anaconda.com/miniconda"

  MAJOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 1)

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

  # Work around bug where devtoolset replaces sudo and breaks it.
  if [ -n "$DEVTOOLSET_VERSION" ]; then
    SUDO=/bin/sudo
  else
    SUDO=sudo
  fi

  as_jenkins() {
    # NB: unsetting the environment variables works around a conda bug
    # https://github.com/conda/conda/issues/6576
    # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
    # NB: This must be run from a directory that jenkins has access to,
    # works around https://github.com/conda/conda-package-handling/pull/34
    $SUDO -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" $*
  }

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  chmod +x "${CONDA_FILE}"
  as_jenkins ./"${CONDA_FILE}" -b -f -p "/opt/conda"
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
  as_jenkins conda install -y python="$ANACONDA_PYTHON_VERSION"

  conda_install() {
    # Ensure that the install command don't upgrade/downgrade Python
    # This should be called as
    #   conda_install pkg1 pkg2 ... [-c channel]
    as_jenkins conda install -q -y python="$ANACONDA_PYTHON_VERSION" $*
  }

  pip_install() {
    as_jenkins pip install --progress-bar off $*
  }

  # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
  CONDA_COMMON_DEPS="astunparse pyyaml mkl=2022.0.1 mkl-include=2022.0.1 setuptools cffi future six"
  if [ "$ANACONDA_PYTHON_VERSION" = "3.10" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy=1.21.2 ${CONDA_COMMON_DEPS} llvmdev=8.0.0
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.9" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy=1.19.2 ${CONDA_COMMON_DEPS} llvmdev=8.0.0
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.8" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy=1.18.5 ${CONDA_COMMON_DEPS} llvmdev=8.0.0
  else
    # Install `typing_extensions` for 3.7
    conda_install numpy=1.18.5 ${CONDA_COMMON_DEPS} typing_extensions
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
