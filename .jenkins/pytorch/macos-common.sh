#!/bin/bash

# Common prelude for macos-build.sh and macos-test.sh

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sysctl -a | grep machdep.cpu

# NOTE: mkl 2021.3.0+ cmake requires sub-command PREPEND, may break the build
retry conda install -y \
  mkl=2021.2.0 \
  mkl-include=2021.2.0 \
  numpy \
  pyyaml=5.3 \
  setuptools \
  cmake=3.19 \
  cffi \
  ninja \
  typing_extensions \
  dataclasses \
  pip

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
