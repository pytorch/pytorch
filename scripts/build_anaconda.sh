#!/bin/bash
#

set -ex

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CONDA_BUILD_ARGS=()

# Build for Python 3.6
# Specifically 3.6 because the latest Anaconda version is 3.6, and so it's site
# packages have 3.6 in the name
PYTHON_FULL_VERSION="$(python --version 2>&1)"
if [[ "$PYTHON_FULL_VERSION" == *3.6* ]]; then
  CONDA_BUILD_ARGS+=(" --python 3.6")
fi

# openmpi is only available in conda-forge (for linux), so conda-forge has to
# be added as a channel for this 'full' build. This causes the default opencv
# to be pulled from conda-forge, which will break with a "can't find
# libopencv_highgui.so", so we also pin opencv version to 3.3.0 to avoid that
# issue
#if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
#  CONDA_BUILD_ARGS+=(" -c conda-forge")
#fi

# Reinitialize submodules
git submodule update --init

# Pick correct conda-build folder
CAFFE2_CONDA_BUILD_DIR="${CAFFE2_ROOT}/conda"
if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda_full"
elif [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda"
else
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/no_cuda"
fi

# Change the package name for CUDA builds to have the specific CUDA and cuDNN
# version in them
CAFFE2_PACKAGE_NAME="caffe2"
if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  # Build name of package
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-cuda${CAFFE2_CUDA_VERSION}-cudnn${CAFFE2_CUDNN_VERSION}"
  if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
    CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-full"
  fi

  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  # WARNING: This does not work on mac.
  sed -i "s/caffe2-cuda\$/${CAFFE2_PACKAGE_NAME}/" "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
fi

# If skipping tests, remove the test related lines from the meta.yaml
if [ -n "$SKIP_CONDA_TESTS" ]; then

  if [ "$(uname)" == 'Darwin' ]; then
    sed -i '' '/test:/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
    sed -i '' '/imports:/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
    sed -i '' '/caffe2.python.core/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
  else
    sed -i '/test:/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
    sed -i '/imports:/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
    sed -i '/caffe2.python.core/d' "${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
  fi

elif [ -n "$UPLOAD_TO_CONDA" ]; then
  # Upload to Anaconda.org if needed. This is only allowed if testing is
  # enabled
  CONDA_BUILD_ARGS+=(" --user ${ANACONDA_USERNAME}")
  CONDA_BUILD_ARGS+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")
fi

# Build Caffe2 with conda-build
# If --user and --token are set, then this will also upload the built package
# to Anaconda.org, provided there were no failures and all the tests passed
conda build "${CAFFE2_CONDA_BUILD_DIR}" ${CONDA_BUILD_ARGS[@]} "$@"

# Install Caffe2 from the built package into the local conda environment
if [ -n "$CONDA_INSTALL_LOCALLY" ]; then
  conda install -y "${CAFFE2_PACKAGE_NAME}" --use-local
fi
