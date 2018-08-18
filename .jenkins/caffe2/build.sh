#!/bin/bash

set -ex

pip install --user --no-cache-dir hypothesis==3.59.0


# The INSTALL_PREFIX here must match up with test.sh
INSTALL_PREFIX="/usr/local/caffe2"
LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
CMAKE_ARGS=()
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
    for compiler in cc c++ gcc g++ x86_64-linux-gnu-gcc; do
      (
        echo "#!/bin/sh"
        echo "exec $SCCACHE $(which $compiler) \"\$@\""
      ) > "./sccache/$compiler"
      chmod +x "./sccache/$compiler"
    done

    if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]]; then
      (
        echo "#!/bin/sh"
        echo "exec $SCCACHE $(which nvcc) \"\$@\""
      ) > "./sccache/nvcc"
      chmod +x "./sccache/nvcc"
    fi

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

report_compile_cache_stats() {
  if [[ -n "${SCCACHE}" ]]; then
    "$SCCACHE" --show-stats
  elif which ccache > /dev/null; then
    ccache -s
  fi
}

###############################################################################
# Explicitly set Python executable.
###############################################################################
# On Ubuntu 16.04 the default Python is still 2.7.
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON=$(which "python${BASH_REMATCH[1]}")
  CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=${PYTHON}")
fi


###############################################################################
# Use special scripts for Android, conda, and setup builds
###############################################################################
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  CMAKE_ARGS+=("-DBUILD_BINARY=ON")
  CMAKE_ARGS+=("-DBUILD_TEST=ON")
  CMAKE_ARGS+=("-DUSE_OBSERVERS=ON")
  CMAKE_ARGS+=("-DUSE_ZSTD=ON")
  "${ROOT_DIR}/scripts/build_android.sh" ${CMAKE_ARGS[*]} "$@"
  exit 0
elif [[ "${BUILD_ENVIRONMENT}" == conda* ]]; then
  "${ROOT_DIR}/scripts/build_anaconda.sh" --skip-tests --install-locally "$@"
  report_compile_cache_stats

  # This build will be tested against onnx tests, which needs onnx installed.
  # At this point the visible protbuf installation will be in conda, since one
  # of Caffe2's dependencies uses conda, so the correct protobuf include
  # headers are those in conda as well
  # This path comes from install_anaconda.sh which installs Anaconda into the
  # docker image
  PROTOBUF_INCDIR=/opt/conda/include pip install -b /tmp/pip_install_onnx "file://${ROOT_DIR}/third_party/onnx#egg=onnx"
  report_compile_cache_stats
  exit 0
elif [[ $BUILD_ENVIRONMENT == *setup* ]]; then
  rm -rf $INSTALL_PREFIX && mkdir $INSTALL_PREFIX
  PYTHONPATH=$INSTALL_PREFIX $PYTHON setup_caffe2.py develop --install-dir $INSTALL_PREFIX
  exit 0
fi


###############################################################################
# Set cmake args
###############################################################################
CMAKE_ARGS+=("-DBUILD_BINARY=ON")
CMAKE_ARGS+=("-DBUILD_TEST=ON")
CMAKE_ARGS+=("-DINSTALL_TEST=ON")
CMAKE_ARGS+=("-DUSE_OBSERVERS=ON")
CMAKE_ARGS+=("-DUSE_ZSTD=ON")
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}")

if [[ $BUILD_ENVIRONMENT == *mkl* ]]; then
  CMAKE_ARGS+=("-DBLAS=MKL")
fi
if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  CMAKE_ARGS+=("-DUSE_CUDA=ON")
  CMAKE_ARGS+=("-DCUDA_ARCH_NAME=Maxwell")
  CMAKE_ARGS+=("-DUSE_NNPACK=OFF")

  # Explicitly set path to NVCC such that the symlink to ccache or sccache is used
  CMAKE_ARGS+=("-DCUDA_NVCC_EXECUTABLE=${CACHE_WRAPPER_DIR}/nvcc")

  # Ensure FindCUDA.cmake can infer the right path to the CUDA toolkit.
  # Setting PATH to resolve to the right nvcc alone isn't enough.
  # See /usr/share/cmake-3.5/Modules/FindCUDA.cmake, block at line 589.
  export CUDA_PATH="/usr/local/cuda"

  # Ensure the ccache symlink can still find the real nvcc binary.
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
  # TODO: This is patching the official FindHip to properly handly
  # cmake generator expression. A PR is opened in the upstream repo here:
  # https://github.com/ROCm-Developer-Tools/HIP/pull/516
  # remove this hack once it's merged.
  if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
    sudo sed -i 's/\ -I${dir}/\ $<$<BOOL:${dir}>:-I${dir}>/' /opt/rocm/hip/cmake/FindHIP.cmake
  fi

  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8
  export HCC_AMDGPU_TARGET=gfx900

  ########## HIPIFY Caffe2 operators
  ${PYTHON} "${ROOT_DIR}/tools/amd_build/build_pytorch_amd.py"
  ${PYTHON} "${ROOT_DIR}/tools/amd_build/build_caffe2_amd.py"
fi

# Try to include Redis support for Linux builds
if [ "$(uname)" == "Linux" ]; then
  CMAKE_ARGS+=("-DUSE_REDIS=ON")
fi

# Currently, on Jenkins mac os, we will use custom protobuf. Mac OS
# contbuild at the moment is minimal dependency - it doesn't use glog
# or gflags either.
if [ "$(uname)" == "Darwin" ]; then
  CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF=ON")
fi

# Use a speciallized onnx namespace in CI to catch hardcoded onnx namespace
CMAKE_ARGS+=("-DONNX_NAMESPACE=ONNX_NAMESPACE_FOR_C2_CI")

# We test the presence of cmake3 (for platforms like Centos and Ubuntu 14.04)
# and use that if so.
if [[ -x "$(command -v cmake3)" ]]; then
    CMAKE_BINARY=cmake3
else
    CMAKE_BINARY=cmake
fi
# sccache will fail for CUDA builds if all cores are used for compiling
if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]] && [ -n "${SCCACHE}" ]; then
  MAX_JOBS=`expr $(nproc) - 1`
else
  MAX_JOBS=$(nproc)
fi


###############################################################################
# Configure and make
###############################################################################

if [[ -z "$INTEGRATED" ]]; then

  # Run cmake from ./build_caffe2 directory so it doesn't conflict with
  # standard PyTorch build directory. Eventually these won't need to
  # be separate.
  rm -rf build_caffe2
  mkdir build_caffe2
  cd ./build_caffe2

  # Configure
  ${CMAKE_BINARY} "${ROOT_DIR}" ${CMAKE_ARGS[*]} "$@"

  # Build
  if [ "$(uname)" == "Linux" ]; then
    make "-j${MAX_JOBS}" install
  else
    echo "Don't know how to build on $(uname)"
    exit 1
  fi

else

  FULL_CAFFE2=1 python setup.py install --user
  # TODO: I'm not sure why this is necessary
  cp -r torch/lib/tmp_install $INSTALL_PREFIX

fi

report_compile_cache_stats


###############################################################################
# Install ONNX
###############################################################################

# Install ONNX into a local directory
pip install --user -b /tmp/pip_install_onnx "file://${ROOT_DIR}/third_party/onnx#egg=onnx"

report_compile_cache_stats

# Symlink the caffe2 base python path into the system python path,
# so that we can import caffe2 without having to change $PYTHONPATH.
# Run in a subshell to contain environment set by /etc/os-release.
#
# This is only done when running on Jenkins!  We don't want to pollute
# the user environment with Python symlinks and ld.so.conf.d hacks.
#
if [[ -z "$INTEGRATED" ]]; then
  if [ -n "${JENKINS_URL}" ]; then
    (
      source /etc/os-release

      function python_version() {
        "$PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[0:2])'
      }

      # Debian/Ubuntu
      if [[ "$ID_LIKE" == *debian* ]]; then
        python_path="/usr/local/lib/$(python_version)/dist-packages"
        sudo ln -sf "${INSTALL_PREFIX}/caffe2" "${python_path}"
      fi

      # RHEL/CentOS
      if [[ "$ID_LIKE" == *rhel* ]]; then
        python_path="/usr/lib64/$(python_version)/site-packages/"
        sudo ln -sf "${INSTALL_PREFIX}/caffe2" "${python_path}"
      fi

      # /etc/ld.so.conf.d is used on both Debian and RHEL
      echo "${INSTALL_PREFIX}/lib" | sudo tee /etc/ld.so.conf.d/caffe2.conf
      sudo ldconfig
    )
  fi
fi
