#!/bin/bash

if [[ "$BUILD_ENVIRONMENT" == "pytorch-linux-xenial-py3-clang5-asan" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" $*
fi

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-build"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

# Target only our CI GPU machine's CUDA arch to speed up the build
#export TORCH_CUDA_ARCH_LIST=5.2

if [[ "$BUILD_ENVIRONMENT" == *trusty-py3.6-gcc5.4* ]]; then
  export DEBUG=1
fi

export USE_SYSTEM_NCCL=1
export NCCL_ROOT_DIR=/usr/local/cuda
export NCCL_LIB_DIR=/usr/local/cuda/lib64
export NCCL_INCLUDE_DIR=/usr/local/cuda/include
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH:/usr/local/cuda/lib64

LD_LIBRARY_PATH=/usr/lib:/usr/local/magma/lib:$LD_LIBRARY_PATH python setup.py install

exit
# Add the test binaries so that they won't be git clean'ed away
git add -f build/bin

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

# Test no-Python build
if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
  echo "Building libtorch"
  # NB: Install outside of source directory (at the same level as the root
  # pytorch folder) so that it doesn't get cleaned away prior to docker push.
  VERBOSE=1 tools/cpp_build/build_caffe2.sh "$PWD/../cpp-build"
fi
