#!/bin/bash

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-test"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

echo "Testing pytorch"

# JIT C++ extensions require ninja.
git clone https://github.com/ninja-build/ninja --quiet
pushd ninja
python ./configure.py --bootstrap
export PATH="$PWD:$PATH"
popd

# DANGER WILL ROBINSON.  The LD_PRELOAD here oculd cause you problems
# if you're not careful.  Check this if you made some changes and the
# ASAN test is not working
if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1
    export PYTORCH_TEST_WITH_ASAN=1
    # TODO: Figure out how to avoid hard-coding these paths
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-5.0/bin/llvm-symbolizer
    export LD_PRELOAD=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux/libclang_rt.asan-x86_64.so
fi

time python test/run_test.py --verbose

# Test ATen
if [[ "$BUILD_ENVIRONMENT" != *asan* ]]; then
  echo "Testing ATen"
  TORCH_LIB_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib
  ln -s "$TORCH_LIB_PATH"/libATen.so aten/build/src/ATen/libATen.so
  aten/tools/run_tests.sh aten/build
fi

rm -rf ninja

echo "Installing torchvision at branch master"
rm -rf vision
# TODO: This git clone is bad
git clone https://github.com/pytorch/vision --quiet
pushd vision
time python setup.py install
popd

if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
   echo "Testing libtorch with NO_PYTHON"
   CPP_BUILD="$PWD/../cpp-build"
   if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
     "$CPP_BUILD"/libtorch/bin/test_jit
   else
     "$CPP_BUILD"/libtorch/bin/test_jit "[cpu]"
   fi
   python tools/download_mnist.py --quiet -d test/cpp/api/mnist
   OMP_NUM_THREADS=2 "$CPP_BUILD"/libtorch/bin/test_api
fi
