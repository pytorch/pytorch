#!/bin/bash
##############################################################################
# Example command to build Caffe2 locally without installing many dependencies
##############################################################################
#
# This script builds protoc locally and then sets the appropriate dependencies
# to remove the need for many external dependencies.
#

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"

# We are going to build the target into build.
BUILD_ROOT=$CAFFE2_ROOT/build
mkdir -p $BUILD_ROOT
echo "Build Caffe2 into: $BUILD_ROOT"

# Build protobuf from third_party so we have a host protoc binary.
echo "Building protoc"
$CAFFE2_ROOT/scripts/build_host_protoc.sh || exit 1

# Now, actually build the target.
echo "Building caffe2"
cd $BUILD_ROOT

cmake .. \
    -DPROTOBUF_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc \
    -DBUILD_SHARED_LIBS=OFF \
    || exit 1
    
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
    cmake --build . -- "-j$(nproc)"
fi
