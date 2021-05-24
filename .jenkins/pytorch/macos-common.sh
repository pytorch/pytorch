#!/bin/bash

# Common prelude for macos-build.sh and macos-test.sh

sysctl -a | grep machdep.cpu

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export PATH="/usr/local/bin:$PATH"
export WORKSPACE_DIR="${HOME}/workspace"
mkdir -p "${WORKSPACE_DIR}"

if [[ "${COMPACT_JOB_NAME}" == *arm64* ]]; then
  MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-MacOSX-x86_64.sh"
else
  MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-MacOSX-x86_64.sh"
fi

# If a local installation of conda doesn't exist, we download and install conda
if [ ! -d "${WORKSPACE_DIR}/miniconda3" ]; then
  mkdir -p "${WORKSPACE_DIR}"
  curl --retry 3 ${MINICONDA_URL} -o "${WORKSPACE_DIR}"/miniconda3.sh
  retry bash "${WORKSPACE_DIR}"/miniconda3.sh -b -p "${WORKSPACE_DIR}"/miniconda3
fi
export PATH="${WORKSPACE_DIR}/miniconda3/bin:$PATH"
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}"/miniconda3/bin/activate
retry conda install -y mkl mkl-include numpy=1.18.5 pyyaml=5.3 setuptools=46.0.0 cmake cffi ninja typing_extensions dataclasses pip
# The torch.hub tests make requests to GitHub.
#
# The certifi package from conda-forge is new enough to make the
# following error disappear (included for future reference):
#
# > ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
# > certificate verify failed: unable to get local issuer certificate
# > (_ssl.c:1056)
#
retry conda install -y -c conda-forge certifi wheel=0.36.2

# Needed by torchvision, which is imported from TestHub in test_utils.py.
retry conda install -y pillow

# Building with USE_DISTRIBUTED=1 requires libuv (for Gloo).
retry conda install -y libuv pkg-config

# Image commit tag is used to persist the build from the build job
# and to retrieve the build from the test job.
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}

# These are required for both the build job and the test job.
# In the latter to test cpp extensions.
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
