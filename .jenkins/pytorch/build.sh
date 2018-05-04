#!/bin/bash

if [[ "$BUILD_ENVIRONMENT" == "pytorch-linux-xenial-py3-clang5-asan" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" $*
fi

# Add nccl2 for distributed test.
apt-get install libnccl-dev libnccl2

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-build"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

# TODO: Don't run this...
pip install -r requirements.txt || true

if ! which conda; then
  pip install mkl mkl-devel
fi

WERROR=1 python setup.py install

# Add the ATen test binaries so that they won't be git clean'ed away
git add -f aten/build/src/ATen/test

# Testing ATen install
if [[ "$BUILD_ENVIRONMENT" != *cuda* ]]; then
  echo "Testing ATen install"
  time tools/test_aten_install.sh
fi

# Test C FFI plugins
# cffi install doesn't work for Python 3.7
if [[ "$BUILD_ENVIRONMENT" != *pynightly* ]]; then
  # TODO: Don't run this here
  pip install cffi
  git clone https://github.com/pytorch/extension-ffi.git
  pushd extension-ffi/script
  python build.py
  popd
fi

# Test documentation build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda8-cudnn6-py3* ]]; then
  pushd docs
  # TODO: Don't run this here
  pip install -r requirements.txt || true
  make html
  popd
fi

# Test no-Python build
if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
  echo "Building libtorch with NO_PYTHON"
  # NB: Install outside of source directory (at the same level as the root
  # pytorch folder) so that it doesn't get cleaned away prior to docker push.
  WERROR=1 VERBOSE=1 tools/cpp_build/build_all.sh "$PWD/../cpp-build"
fi
