#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variables:
#   $JOB_NAME
#   $PYTHON_VERSION
#   $GCC_VERSION

export PATH=/opt/conda/bin:$PATH

if [[ "$JOB_NAME" == *cuda* ]]; then
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   # The ccache wrapper should be able to find the real nvcc
   export PATH="/usr/local/cuda/bin:$PATH"
else
   export PATH=/opt/python/${PYTHON_VERSION}/bin:$PATH
   export LD_LIBRARY_PATH=/opt/python/${PYTHON_VERSION}/lib:$LD_LIBRARY_PATH

   # NB: setup.py chokes on a setting of CC='ccache gcc' (two words),
   # so we created a symlinked binary that we can pass as CC in one word
   mkdir ./ccache
   ln -sf "$(which ccache)" ./ccache/gcc-${GCC_VERSION}
   ln -sf "$(which ccache)" ./ccache/g++-${GCC_VERSION}
   export CC="$PWD/ccache/gcc-${GCC_VERSION}"
   export CXX="$PWD/ccache/g++-${GCC_VERSION}"
fi

echo "Installing torchvision at branch master"
rm -rf vision
git clone https://github.com/pytorch/vision --quiet
if [[ "$JOB_NAME" == *cuda* ]]; then
   conda install -y pillow
else
   pip install pillow
fi

echo "Testing pytorch"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ "$JOB_NAME" == *asan* ]]; then
    export PATH="/usr/lib/llvm-5.0/bin:$PATH"
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1
    export PYTORCH_TEST_WITH_ASAN=1
fi

# JIT C++ extensions require ninja.
git clone https://github.com/ninja-build/ninja --quiet
pushd ninja
python ./configure.py --bootstrap
export PATH="$PWD:$PATH"
popd

if [[ "$JOB_NAME" == *asan* ]]; then
    export LD_PRELOAD=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux/libclang_rt.asan-x86_64.so
fi

time test/run_test.sh -- -v

rm -rf ninja

pushd vision
time python setup.py install
popd
