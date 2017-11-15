#!/bin/bash
#
##############################################################################
# Example command to build Caffe2
##############################################################################
#

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"

# Build protobuf compiler from third_party if configured to do so
if [ -n "${USE_HOST_PROTOC:-}" ]; then
    echo "Building protoc before compiling Caffe2..."
    $CAFFE2_ROOT/scripts/build_host_protoc.sh || exit 1
    CMAKE_ARGS="$CMAKE_ARGS \
        -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc"
fi

# We are going to build the target into build.
BUILD_ROOT="$CAFFE2_ROOT/build"
mkdir -p "$BUILD_ROOT"
echo "Building Caffe2 in: $BUILD_ROOT"

# Now, actually build the target.
cd "$BUILD_ROOT"
set -x
cmake .. ${CMAKE_ARGS} "$@" || exit 1

if [ "$(uname)" == 'Darwin' ]; then
  # Use ccache if available (this path is where Homebrew installs ccache symlinks)
  if [ -d /usr/local/opt/ccache/libexec ]; then
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
  fi

  cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
  cmake --build . -- "-j$(nproc)"
fi
