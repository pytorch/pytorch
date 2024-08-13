#!/bin/bash
##############################################################################
# Example command to build the mobile target.
##############################################################################
#
# This script shows how one can build a libtorch library optimized for mobile
# devices using host toolchain.

set -e

export BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN=1
CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

echo "Bash: $(/bin/bash --version | head -1)"
echo "Caffe2 path: $CAFFE2_ROOT"

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')")
CMAKE_ARGS+=("-DPython_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')")
CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF=OFF")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=OFF")

# custom build with selected ops
if [ -n "${SELECTED_OP_LIST}" ]; then
  SELECTED_OP_LIST="$(cd $(dirname $SELECTED_OP_LIST); pwd -P)/$(basename $SELECTED_OP_LIST)"
  echo "Choose SELECTED_OP_LIST file: $SELECTED_OP_LIST"
  if [ ! -r ${SELECTED_OP_LIST} ]; then
    echo "Error: SELECTED_OP_LIST file ${SELECTED_OP_LIST} not found."
    exit 1
  fi
  CMAKE_ARGS+=("-DSELECTED_OP_LIST=${SELECTED_OP_LIST}")
fi

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

# Don't build artifacts we don't need
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DBUILD_BINARY=OFF")

# If there exists env variable and it equals to 1, build lite interpreter.
# Default behavior is to build full jit interpreter.
# cmd:  BUILD_LITE_INTERPRETER=1 ./scripts/build_mobile.sh
if [ "x${BUILD_LITE_INTERPRETER}" == "x1" ]; then
  CMAKE_ARGS+=("-DBUILD_LITE_INTERPRETER=ON")
else
  CMAKE_ARGS+=("-DBUILD_LITE_INTERPRETER=OFF")
fi
if [ "x${TRACING_BASED}" == "x1" ]; then
  CMAKE_ARGS+=("-DTRACING_BASED=ON")
else
  CMAKE_ARGS+=("-DTRACING_BASED=OFF")
fi

# Lightweight dispatch bypasses the PyTorch Dispatcher.
if [ "${USE_LIGHTWEIGHT_DISPATCH}" == 1 ]; then
  CMAKE_ARGS+=("-DUSE_LIGHTWEIGHT_DISPATCH=ON")
  CMAKE_ARGS+=("-DSTATIC_DISPATCH_BACKEND=CPU")
else
  CMAKE_ARGS+=("-DUSE_LIGHTWEIGHT_DISPATCH=OFF")
fi

# Disable unused dependencies
CMAKE_ARGS+=("-DUSE_ROCM=OFF")
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_ITT=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")
CMAKE_ARGS+=("-DUSE_OPENMP=OFF")
CMAKE_ARGS+=("-DUSE_ONEDNN=OFF")
CMAKE_ARGS+=("-DUSE_NNPACK=OFF")
CMAKE_ARGS+=("-DUSE_NUMPY=OFF")
CMAKE_ARGS+=("-DUSE_BLAS=OFF")

# Only toggle if VERBOSE=1
if [ "${VERBOSE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
fi

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=("$@")

# Now, actually build the Android target.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_mobile"}
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT
cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ -z "$MAX_JOBS" ]; then
  if [ "$(uname)" == 'Darwin' ]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
  else
    MAX_JOBS=$(nproc)
  fi
fi

echo "Will install headers and libs to $INSTALL_PREFIX for further project usage."
cmake --build . --target install -- "-j${MAX_JOBS}"
echo "Installation completed, now you can copy the headers/libs from $INSTALL_PREFIX to your project directory."
