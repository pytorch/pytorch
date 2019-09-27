#!/bin/bash
##############################################################################
# Example command to build the mobile target.
##############################################################################
#
# This script shows how one can build a libtorch library optimized for mobile
# devices using host toolchain.

set -e

export PYTORCH_BUILD_MOBILE=1
CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

echo "Bash: $(/bin/bash --version | head -1)"
echo "Caffe2 path: $CAFFE2_ROOT"

# Now, actually build the Android target.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_mobile"}
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT

CMAKE_ARGS=()
CMAKE_ARGS+=("-DBUILD_CAFFE2_MOBILE=OFF")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')")
CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')")
CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF=OFF")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=OFF")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

# Disable unused dependencies
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_LMDB=OFF")
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")
CMAKE_ARGS+=("-DUSE_OPENMP=OFF")

# Only toggle if VERBOSE=1
if [ "${VERBOSE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
fi

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

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
