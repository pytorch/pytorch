#!/bin/bash

set -ex

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# CMAKE_ARGS are only passed to 'cmake' and the -Dfoo=bar does not work with
# setup.py, so we build a list of foo=bars and then either convert it to
# -Dfoo=bars or export them before running setup.py
build_args=()
build_to_cmake () {
  cmake_args=()
  for build_arg in $*; do
    cmake_args+=("-D$build_arg")
  done
  echo ${cmake_args[@]}
}


SCCACHE="$(which sccache)"

# Setup ccache if configured to use it (and not sccache)
if [ -z "${SCCACHE}" ] && which ccache > /dev/null; then
  mkdir -p ./ccache
  ln -sf "$(which ccache)" ./ccache/cc
  ln -sf "$(which ccache)" ./ccache/c++
  ln -sf "$(which ccache)" ./ccache/gcc
  ln -sf "$(which ccache)" ./ccache/g++
  ln -sf "$(which ccache)" ./ccache/x86_64-linux-gnu-gcc
  if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]]; then
    ln -sf "$(which ccache)" ./ccache/nvcc
  fi
  export CACHE_WRAPPER_DIR="$PWD/ccache"
  export PATH="$CACHE_WRAPPER_DIR:$PATH"
fi

# sccache will fail for CUDA builds if all cores are used for compiling
if [ -z "$MAX_JOBS" ]; then
  if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]] && [ -n "${SCCACHE}" ]; then
    MAX_JOBS=`expr $(nproc) - 1`
  else
    MAX_JOBS=$(nproc)
  fi
fi

report_compile_cache_stats() {
  if [[ -n "${SCCACHE}" ]]; then
    "$SCCACHE" --show-stats
  elif which ccache > /dev/null; then
    ccache -s
  fi
}


###############################################################################
# Use special scripts for Android and setup builds
###############################################################################
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  build_args+=("BUILD_BINARY=ON")
  build_args+=("BUILD_TEST=ON")
  build_args+=("USE_OBSERVERS=ON")
  build_args+=("USE_ZSTD=ON")
  BUILD_CAFFE2_MOBILE=1 "${ROOT_DIR}/scripts/build_android.sh" $(build_to_cmake ${build_args[@]}) "$@"
  exit 0
fi

###############################################################################
# Set parameters
###############################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  build_args+=("BUILD_PYTHON=OFF")
else
  build_args+=("BUILD_PYTHON=ON")
  build_args+=("PYTHON_EXECUTABLE=${PYTHON}")
fi
if [[ $BUILD_ENVIRONMENT == *mkl* ]]; then
  build_args+=("BLAS=MKL")
  build_args+=("USE_MKLDNN=ON")
fi
build_args+=("BUILD_BINARY=ON")
build_args+=("BUILD_TEST=ON")
build_args+=("INSTALL_TEST=ON")
build_args+=("USE_ZSTD=ON")

if [[ $BUILD_ENVIRONMENT == *py2-cuda9.0-cudnn7-ubuntu16.04* ]]; then
  # removing http:// duplicate in favor of nvidia-ml.list
  # which is https:// version of the same repo
  sudo rm -f /etc/apt/sources.list.d/nvidia-machine-learning.list
  curl --retry 3 -o ./nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb
  sudo dpkg -i ./nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb
  sudo apt-key add /var/nvinfer-runtime-trt-repo-5.0.2-ga-cuda9.0/7fa2af80.pub
  sudo apt-get -qq update
  sudo apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda9.0 libnvinfer-dev=5.0.2-1+cuda9.0
  rm ./nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb

  build_args+=("USE_TENSORRT=ON")
fi

if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  build_args+=("USE_CUDA=ON")
  build_args+=("USE_NNPACK=OFF")

  # Target only our CI GPU machine's CUDA arch to speed up the build
  build_args+=("TORCH_CUDA_ARCH_LIST=Maxwell")

  # Explicitly set path to NVCC such that the symlink to ccache or sccache is used
  if [ -n "${CACHE_WRAPPER_DIR}" ]; then
    build_args+=("CUDA_NVCC_EXECUTABLE=${CACHE_WRAPPER_DIR}/nvcc")
  fi

  # Ensure FindCUDA.cmake can infer the right path to the CUDA toolkit.
  # Setting PATH to resolve to the right nvcc alone isn't enough.
  # See /usr/share/cmake-3.5/Modules/FindCUDA.cmake, block at line 589.
  export CUDA_PATH="/usr/local/cuda"

  # Ensure the ccache symlink can still find the real nvcc binary.
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
  if [[ -n "$IN_CI" && -z "$PYTORCH_ROCM_ARCH" ]]; then
      # Set ROCM_ARCH to gfx900 and gfx906 for CI builds, if user doesn't override.
      echo "Limiting PYTORCH_ROCM_ARCH to gfx90[06] for CI builds"
      export PYTORCH_ROCM_ARCH="gfx900;gfx906"
  fi
  # This is needed to enable ImageInput operator in resnet50_trainer
  build_args+=("USE_OPENCV=ON")
  # This is needed to read datasets from https://download.caffe2.ai/databases/resnet_trainer.zip
  build_args+=("USE_LMDB=ON")
  # hcc used to run out of memory, silently exiting without stopping
  # the build process, leaving undefined symbols in the shared lib,
  # causing undefined symbol errors when later running tests.
  # We used to set MAX_JOBS to 4 to avoid, but this is no longer an issue.
  if [ -z "$MAX_JOBS" ]; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi

  ########## HIPIFY Caffe2 operators
  ${PYTHON} "${ROOT_DIR}/tools/amd_build/build_amd.py"
fi

# Try to include Redis support for Linux builds
if [ "$(uname)" == "Linux" ]; then
  build_args+=("USE_REDIS=ON")
fi

# Use a specialized onnx namespace in CI to catch hardcoded onnx namespace
build_args+=("ONNX_NAMESPACE=ONNX_NAMESPACE_FOR_C2_CI")

###############################################################################
# Configure and make
###############################################################################

if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  # cmake-only non-setup.py build, to test cpp only bits. This installs into
  # /usr/local/caffe2 and installs no Python tests
  build_args+=("CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}")

  # Run cmake from ./build_caffe2 directory so it doesn't conflict with
  # standard PyTorch build directory. Eventually these won't need to
  # be separate.
  rm -rf build_caffe2
  mkdir build_caffe2
  cd ./build_caffe2

  # We test the presence of cmake3 (for platforms like Centos and Ubuntu 14.04)
  # and use that if so.
  if [[ -x "$(command -v cmake3)" ]]; then
      CMAKE_BINARY=cmake3
  else
      CMAKE_BINARY=cmake
  fi

  # Configure
  ${CMAKE_BINARY} "${ROOT_DIR}" $(build_to_cmake ${build_args[@]}) "$@"

  # Build
  if [ "$(uname)" == "Linux" ]; then
    make "-j${MAX_JOBS}" install
  else
    echo "Don't know how to build on $(uname)"
    exit 1
  fi

  # This is to save test binaries for testing
  mv "$INSTALL_PREFIX/test/" "$INSTALL_PREFIX/cpp_test/"

  ls -lah $INSTALL_PREFIX

else
  # Python build. Uses setup.py to install into site-packages
  build_args+=("USE_LEVELDB=ON")
  build_args+=("USE_LMDB=ON")
  build_args+=("USE_OPENCV=ON")
  build_args+=("BUILD_TEST=ON")
  # These flags preserve the flags that were used before this refactor (blame
  # me)
  build_args+=("USE_GLOG=ON")
  build_args+=("USE_GFLAGS=ON")
  build_args+=("USE_FBGEMM=OFF")
  build_args+=("USE_MKLDNN=OFF")
  build_args+=("USE_DISTRIBUTED=ON")
  for build_arg in "${build_args[@]}"; do
    export $build_arg
  done

  # sccache will be stuck if  all cores are used for compiling
  # see https://github.com/pytorch/pytorch/pull/7361
  if [[ -n "${SCCACHE}" && $BUILD_ENVIRONMENT != *rocm* ]]; then
    export MAX_JOBS=`expr $(nproc) - 1`
  fi

  pip install --user dataclasses typing_extensions

  $PYTHON setup.py install --user

  report_compile_cache_stats
fi

###############################################################################
# Install ONNX
###############################################################################

# Install ONNX into a local directory
pip install --user "file://${ROOT_DIR}/third_party/onnx#egg=onnx"

report_compile_cache_stats

if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
  # remove sccache wrappers post-build; runtime compilation of MIOpen kernels does not yet fully support them
  sudo rm -f /opt/cache/bin/cc
  sudo rm -f /opt/cache/bin/c++
  sudo rm -f /opt/cache/bin/gcc
  sudo rm -f /opt/cache/bin/g++
  pushd /opt/rocm/llvm/bin
  if [[ -d original ]]; then
    sudo mv original/clang .
    sudo mv original/clang++ .
  fi
  sudo rm -rf original
  popd
fi
