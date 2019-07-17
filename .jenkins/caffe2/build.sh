#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# TODO: Migrate all centos jobs to use proper devtoolset
if [[ "$BUILD_ENVIRONMENT" == *py2-cuda9.0-cudnn7-centos7* ]]; then
  # There is a bug in pango packge on Centos7 that causes undefined
  # symbols, upgrading glib2 to >=2.56.1 solves the issue. See
  # https://bugs.centos.org/view.php?id=15495
  sudo yum install -y -q glib2-2.56.1
fi

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
if [ "$(which gcc)" != "/root/sccache/gcc" ]; then
  # Setup SCCACHE
  ###############################################################################
  # Setup sccache if SCCACHE_BUCKET is set
  if [ -n "${SCCACHE_BUCKET}" ]; then
    mkdir -p ./sccache

    SCCACHE="$(which sccache)"
    if [ -z "${SCCACHE}" ]; then
      echo "Unable to find sccache..."
      exit 1
    fi

    # Setup wrapper scripts
    wrapped="cc c++ gcc g++ x86_64-linux-gnu-gcc"
    if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]]; then
        wrapped="$wrapped nvcc"
    fi
    for compiler in $wrapped; do
      (
        echo "#!/bin/sh"

        # TODO: if/when sccache gains native support for an
        # SCCACHE_DISABLE flag analogous to ccache's CCACHE_DISABLE,
        # this can be removed. Alternatively, this can be removed when
        # https://github.com/pytorch/pytorch/issues/13362 is fixed.
        #
        # NOTE: carefully quoted - we want `which compiler` to be
        # resolved as we execute the script, but SCCACHE_DISABLE and
        # $@ to be evaluated when we execute the script
        echo 'test $SCCACHE_DISABLE && exec '"$(which $compiler)"' "$@"'

        echo "exec $SCCACHE $(which $compiler) \"\$@\""
      ) > "./sccache/$compiler"
      chmod +x "./sccache/$compiler"
    done

    export CACHE_WRAPPER_DIR="$PWD/sccache"

    # CMake must find these wrapper scripts
    export PATH="$CACHE_WRAPPER_DIR:$PATH"
  fi
fi

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
  "${ROOT_DIR}/scripts/build_android.sh" $(build_to_cmake ${build_args[@]}) "$@"
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
  curl -o ./nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0_1-1_amd64.deb
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
  build_args+=("CUDA_NVCC_EXECUTABLE=${CACHE_WRAPPER_DIR}/nvcc")

  # Ensure FindCUDA.cmake can infer the right path to the CUDA toolkit.
  # Setting PATH to resolve to the right nvcc alone isn't enough.
  # See /usr/share/cmake-3.5/Modules/FindCUDA.cmake, block at line 589.
  export CUDA_PATH="/usr/local/cuda"

  # Ensure the ccache symlink can still find the real nvcc binary.
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
  build_args+=("USE_ROCM=ON")
  # This is needed to enable ImageInput operator in resnet50_trainer
  build_args+=("USE_OPENCV=ON")
  # This is needed to read datasets from https://download.caffe2.ai/databases/resnet_trainer.zip
  build_args+=("USE_LMDB=ON")
  # When hcc runs out of memory, it silently exits without stopping
  # the build process, leaving undefined symbols in the shared lib
  # which will cause undefined symbol errors when later running
  # tests. Setting MAX_JOBS to smaller number to make CI less flaky.
  export MAX_JOBS=4

  ########## HIPIFY Caffe2 operators
  ${PYTHON} "${ROOT_DIR}/tools/amd_build/build_amd.py"
fi

# building bundled nccl in this config triggers a bug in nvlink. For
# more, see https://github.com/pytorch/pytorch/issues/14486
if [[ "${BUILD_ENVIRONMENT}" == *-cuda8*-cudnn7* ]]; then
    build_args+=("USE_SYSTEM_NCCL=ON")
fi

# Try to include Redis support for Linux builds
if [ "$(uname)" == "Linux" ]; then
  build_args+=("USE_REDIS=ON")
fi

# Use a speciallized onnx namespace in CI to catch hardcoded onnx namespace
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
  if [[ -n "${SCCACHE}" ]]; then
    export MAX_JOBS=`expr $(nproc) - 1`
  fi

  $PYTHON setup.py install --user

  report_compile_cache_stats
fi

###############################################################################
# Install ONNX
###############################################################################

# Install ONNX into a local directory
pip install --user -b /tmp/pip_install_onnx "file://${ROOT_DIR}/third_party/onnx#egg=onnx"

if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
  ORIG_COMP=/opt/rocm/hcc/bin/clang-*_original
  if [ -e $ORIG_COMP ]; then
    # runtime compilation of MIOpen kernels manages to crash sccache - hence undo the wrapping
    # note that the wrapping always names the compiler "clang-7.0_original"
    WRAPPED=/opt/rocm/hcc/bin/clang-[0-99]
    sudo mv $ORIG_COMP $WRAPPED
  fi
fi

report_compile_cache_stats
