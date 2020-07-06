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

  mkdir /opt/conda
  chown jenkins:jenkins /opt/conda

  as_jenkins() {
    # NB: unsetting the environment variables works around a conda bug
    # https://github.com/conda/conda/issues/6576
    # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
    # NB: This must be run from a directory that jenkins has access to,
    # works around https://github.com/conda/conda-package-handling/pull/34
    sudo -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" $*
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

  # Track latest conda update
  as_jenkins conda update -n base conda

  # Install correct Python version
  as_jenkins conda install python="$ANACONDA_PYTHON_VERSION"

  conda_install() {
    # Ensure that the install command don't upgrade/downgrade Python
    # This should be called as
    #   conda_install pkg1 pkg2 ... [-c channel]
    as_jenkins conda install -q -y python="$ANACONDA_PYTHON_VERSION" $*
  }

  # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
  # DO NOT install cmake here as it would install a version newer than 3.5, but
  # we want to pin to version 3.5.
  if [ "$ANACONDA_PYTHON_VERSION" = "3.8" ]; then
    # DO NOT install typing if installing python-3.8, since its part of python-3.8 core packages
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    conda_install numpy pyyaml mkl mkl-include setuptools cffi future six llvmdev=8.0.0
  else
    conda_install numpy pyyaml mkl mkl-include setuptools cffi typing future six
  fi
  if [[ "$CUDA_VERSION" == 9.2* ]]; then
    conda_install magma-cuda92 -c pytorch
  elif [[ "$CUDA_VERSION" == 10.0* ]]; then
    conda_install magma-cuda100 -c pytorch
  elif [[ "$CUDA_VERSION" == 10.1* ]]; then
    conda_install magma-cuda101 -c pytorch
  elif [[ "$CUDA_VERSION" == 10.2* ]]; then
    conda_install magma-cuda102 -c pytorch
  fi

  # TODO: This isn't working atm
  conda_install nnpack -c killeent

  # Install some other packages
  # TODO: Why is scipy pinned
  # numba & llvmlite is pinned because of https://github.com/numba/numba/issues/4368
  # scikit-learn is pinned because of
  # https://github.com/scikit-learn/scikit-learn/issues/14485 (affects gcc 5.5
  # only)
  as_jenkins pip install --progress-bar off pytest scipy==1.1.0 scikit-learn==0.20.3 scikit-image librosa>=0.6.2 psutil numba==0.46.0 llvmlite==0.30.0

  popd
fi
