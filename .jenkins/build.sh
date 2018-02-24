#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variables:
#   $JOB_NAME
#   $PYTHON_VERSION
#   $GCC_VERSION
#
# TODO: change this script to make use of $BUILD_ENVIRONMENT,
# which we can hard code into Docker images and then these scripts
# will work out of the box without having to set any env vars.

export PATH=/opt/conda/bin:$PATH

if [[ "$JOB_NAME" != *cuda* ]]; then
   export PATH=/opt/python/${PYTHON_VERSION}/bin:$PATH
   export LD_LIBRARY_PATH=/opt/python/${PYTHON_VERSION}/lib:$LD_LIBRARY_PATH
   export CC="sccache /usr/bin/gcc-${GCC_VERSION}"
   export CXX="sccache /usr/bin/g++-${GCC_VERSION}"

   sccache --show-stats

   function sccache_epilogue() {
      sccache --show-stats
   }
   trap_add sccache_epilogue EXIT

else
   # CMake should use ccache symlink for nvcc
   export CUDA_NVCC_EXECUTABLE=/usr/local/bin/nvcc
   # The ccache wrapper should be able to find the real nvcc
   export PATH=/usr/local/cuda/bin:$PATH

   # Add CUDA stub/real path to loader path
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   # Build for Maxwell
   export TORCH_CUDA_ARCH_LIST="Maxwell"
   export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
fi

echo "Python Version:"
which python
python --version

pip install -r requirements.txt || true

if [[ "$JOB_NAME" == *asan* ]]; then
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1
    # Disable Valgrind tests in run_aten_tests.sh; otherwise
    # we'll be valgrind'ing an ASAN'ed binary!  ASANity.
    export VALGRIND=0

    sudo apt-get update
    sudo apt-get install clang-5.0

    export PATH="/usr/lib/llvm-5.0/bin:$PATH"

    # TODO: Figure out how to avoid hard-coding these paths
    LD_LIBRARY_PATH=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux \
      CC="sccache clang" \
      CXX="sccache clang++" \
      LDSHARED="clang --shared" \
      LDFLAGS="-stdlib=libstdc++" \
      CFLAGS="-fsanitize=address -shared-libasan" \
      NO_CUDA=1 \
      python setup.py install

    export LD_PRELOAD=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux/libclang_rt.asan-x86_64.so

else
    python setup.py install

fi

if [[ "$JOB_NAME" != *cuda* ]]; then
   echo "Testing ATen"
   time tools/run_aten_tests.sh
fi

# Test C FFI plugins
# cffi install doesn't work for Python 3.7
if [[ "$JOB_NAME" != *pynightly* ]]; then
   pip install cffi
   git clone https://github.com/pytorch/extension-ffi.git
   cd extension-ffi/script
   python build.py
fi
