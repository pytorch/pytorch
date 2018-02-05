#!/bin/bash
#

set -ex

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CONDA_BLD_ARGS=()

# Build for Python 3.6
# Specifically 3.6 because the latest Anaconda version is 3.6, and so it's site
# packages have 3.6 in the name
PYTHON_FULL_VERSION="$(python --version 2>&1)"
if [[ "$PYTHON_FULL_VERSION" == *3.6* ]]; then
  CONDA_BLD_ARGS+=(" --python 3.6")
fi

# Upload to Anaconda.org if needed
if [ -n "$UPLOAD_TO_CONDA" ]; then
  CONDA_BLD_ARGS+=(" --user ${ANACONDA_USERNAME}")
  CONDA_BLD_ARGS+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")
fi

# Reinitialize submodules
git submodule update --init

# Separate build folder for CUDA builds so that the packages have different
# names
if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  # NOTE: These are magic strings that exist in the meta.yaml
  # WARNING: This does not work on mac.
  sed -i "s/%%CUDA_VERSION%%/${CAFFE2_CUDA_VERSION}/" "${CAFFE2_ROOT}/conda/cuda/meta.yaml"
  sed -i "s/%%CUDNN_VERSION%%/${CAFFE2_CUDNN_VERSION}/" "${CAFFE2_ROOT}/conda/cuda/meta.yaml"

  conda build "${CAFFE2_ROOT}/conda/cuda" ${CONDA_BLD_ARGS[@]} "$@"

  # Change the names back
  sed -i "s/${CAFFE2_CUDA_VERSION}/%%CUDA_VERSION%%/" "${CAFFE2_ROOT}/conda/cuda/meta.yaml"
  sed -i "s/${CAFFE2_CUDNN_VERSION}/%%CUDNN_VERSION%%/" "${CAFFE2_ROOT}/conda/cuda/meta.yaml"
else
  conda build "${CAFFE2_ROOT}/conda/no_cuda" ${CONDA_BLD_ARGS[@]} "$@"
fi
