#!/bin/bash

COMPACT_JOB_NAME=pytorch-macos-10.13-py3-build-test
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export PATH="/usr/local/bin:$PATH"

# Set up conda environment
export PYTORCH_ENV_DIR="${HOME}/pytorch-ci-env"
# If a local installation of conda doesn't exist, we download and install conda
if [ ! -d "${PYTORCH_ENV_DIR}/miniconda3" ]; then
  mkdir -p ${PYTORCH_ENV_DIR}
  curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ${PYTORCH_ENV_DIR}/miniconda3.sh
  bash ${PYTORCH_ENV_DIR}/miniconda3.sh -b -p ${PYTORCH_ENV_DIR}/miniconda3
fi
export PATH="${PYTORCH_ENV_DIR}/miniconda3/bin:$PATH"
source ${PYTORCH_ENV_DIR}/miniconda3/bin/activate
conda install -y mkl mkl-include numpy pyyaml setuptools cmake cffi ninja
rm -rf ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch

git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${PYTORCH_ENV_DIR}/miniconda3/

# Build and test PyTorch
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
# If we run too many parallel jobs, we will OOM
export MAX_JOBS=2

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}

python setup.py install

# Upload torch binaries when the build job is finished
7z a ${IMAGE_COMMIT_TAG}.7z ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch
aws s3 cp ${IMAGE_COMMIT_TAG}.7z s3://ossci-macos-build/pytorch/${IMAGE_COMMIT_TAG}.7z --acl public-read
