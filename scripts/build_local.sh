#!/bin/bash
#
##############################################################################
# Example command to build Caffe2
##############################################################################
#

set -e

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

CMAKE_ARGS=()

# Build protobuf compiler from third_party if configured to do so
if [ -n "${USE_HOST_PROTOC:-}" ]; then
  echo "USE_HOST_PROTOC is set; building protoc before building Caffe2..."
  "$CAFFE2_ROOT/scripts/build_host_protoc.sh"
  CUSTOM_PROTOC_EXECUTABLE="$CAFFE2_ROOT/build_host_protoc/bin/protoc"
  echo "Built protoc $("$CUSTOM_PROTOC_EXECUTABLE" --version)"
  CMAKE_ARGS+=("-DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$CUSTOM_PROTOC_EXECUTABLE")
fi

# We are going to build the target into build.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"
echo "Building Caffe2 in: $BUILD_ROOT"

if [ "$(uname)" == 'Darwin' ]; then
  # Use ccache if available (this path is where Homebrew installs ccache symlinks)
  CCACHE_WRAPPER_PATH=/usr/local/opt/ccache/libexec
  if [ -d "$CCACHE_WRAPPER_PATH" ]; then
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$CCACHE_WRAPPER_PATH/gcc")
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CCACHE_WRAPPER_PATH/g++")
  fi
fi

# Now, actually build the target.
cmake "$CAFFE2_ROOT" \
      "${CMAKE_ARGS[@]}" \
      "$@"

if [ "$(uname)" == 'Darwin' ]; then
  cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
  cmake --build . -- "-j$(nproc)"
fi
