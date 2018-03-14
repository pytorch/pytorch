#!/bin/bash

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-build"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

export PATH=/opt/conda/bin:$PATH

if [[ "$BUILD_ENVIRONMENT" != *cuda* ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/common-linux-cpu.sh"

  export PATH=/opt/python/${TRAVIS_PYTHON_VERSION}/bin:$PATH
  export LD_LIBRARY_PATH=/opt/python/${TRAVIS_PYTHON_VERSION}/lib:$LD_LIBRARY_PATH
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

if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
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

if [[ "$BUILD_ENVIRONMENT" != *cuda* ]]; then
  echo "Testing ATen"
  time tools/run_aten_tests.sh
fi

# Test C FFI plugins
# cffi install doesn't work for Python 3.7
if [[ "$BUILD_ENVIRONMENT" != *pynightly* ]]; then
  pip install cffi
  git clone https://github.com/pytorch/extension-ffi.git
  pushd extension-ffi/script
  python build.py
  popd
fi

# Test documentation build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda8-cudnn6-py3* ]]; then
  pushd docs
  pip install -r requirements.txt || true
  make html
  popd
fi

# Test no-Python build
if [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda9-cudnn7-py3 ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-trusty-py3.6-gcc7.2 ]]; then
  echo "Building libtorch with NO_PYTHON"
  pushd tools/cpp_build || exit 1
  bash build_all.sh
  popd
fi
