#!/bin/bash

# Common prelude for macos-build.sh and macos-test.sh

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sysctl -a | grep machdep.cpu

if [[ ${BUILD_ENVIRONMENT} = *arm64* ]]; then
  # We use different versions here as the arm build/tests runs on python 3.9
  # while the x86 one runs on python 3.8
  retry conda install -c conda-forge -y \
    numpy=1.22.3 \
    pyyaml=6.0 \
    setuptools=61.2.0 \
    cmake=3.22.1 \
    cffi \
    ninja \
    typing_extensions \
    dataclasses \
    pip
else
  # NOTE: mkl 2021.3.0+ cmake requires sub-command PREPEND, may break the build
  retry conda install -c conda-forge -y \
    mkl=2021.2.0 \
    mkl-include=2021.2.0 \
    numpy=1.18.5 \
    pyyaml=5.3 \
    setuptools=46.0.0 \
    cmake=3.22.1 \
    cffi \
    ninja \
    typing_extensions \
    dataclasses \
    pip
fi

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

# These are required for both the build job and the test job.
# In the latter to test cpp extensions.
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
