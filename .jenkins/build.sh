#!/bin/bash

if [[ "$BUILD_ENVIRONMENT" == "pytorch-linux-xenial-py3-clang5-asan" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" $*
fi

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

python setup.py install

# Test ATen
if [[ "$BUILD_ENVIRONMENT" != *cuda* ]]; then
  echo "Testing ATen"
  time tools/run_aten_tests.sh
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
  # NB: Install outside of source directory so that it doesn't get
  # cleaned away prior to docker push
  LIBTORCH_INSTALL_PREFIX=`pwd`/../libtorch
  pushd tools/cpp_build
  bash build_all.sh "$LIBTORCH_INSTALL_PREFIX"
  popd
fi
