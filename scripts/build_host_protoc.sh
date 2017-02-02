#!/bin/bash
##############################################################################
# Build script to build the protoc compiler for the host platform.
##############################################################################
# This script builds the protoc compiler for the host platform, which is needed
# for any cross-compilation as we will need to convert the protobuf source
# files to cc files.
#
# After the execution of the file, one should be able to find the host protoc
# binary at build_host_protoc/bin/protoc.

CAFFE2_ROOT="$( cd "$(dirname -- "$0")"/.. ; pwd -P)"
BUILD_ROOT=$CAFFE2_ROOT/build_host_protoc
mkdir -p $BUILD_ROOT/build

cd $BUILD_ROOT/build
CMAKE=$(which cmake || which /usr/bin/cmake || which /usr/local/bin/cmake)
$CMAKE $CAFFE2_ROOT/third_party/protobuf/cmake \
    -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT \
    -Dprotobuf_BUILD_TESTS=OFF \
    || exit 1
make -j 4 || exit 1
make install || exit 1
