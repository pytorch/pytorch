#!/bin/bash
##############################################################################
# Build script to build the protoc compiler for the host platform.
##############################################################################
# This script builds the protoc compiler for the host platform, which is needed
# for any cross-compilation as we will need to convert the protobuf source
# files to cc files.
#
# --other-flags accepts flags that should be passed to cmake. Optional.
#
# After the execution of the file, one should be able to find the host protoc
# binary at build_host_protoc/bin/protoc.

CAFFE2_ROOT="$( cd "$(dirname -- "$0")"/.. ; pwd -P)"
BUILD_ROOT=$CAFFE2_ROOT/build_host_protoc
mkdir -p $BUILD_ROOT/build

cd $BUILD_ROOT/build
CMAKE=$(which cmake || which /usr/bin/cmake || which /usr/local/bin/cmake)

SHARED="$CAFFE2_ROOT/third_party/protobuf/cmake -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT -Dprotobuf_BUILD_TESTS=OFF "
OTHER_FLAGS=""

while true; do
    case "$1" in
        --other-flags)
            shift;
            echo "Other flags passed to cmake: $@";
            OTHER_FLAGS=" $@ ";
            break ;;
        "")
            break ;;
        *)
            echo "Unknown option passed as argument: $1"
            break ;;
    esac
done


$CMAKE $SHARED $OTHER_FLAGS || exit 1
make -j 4 || exit 1
make install || exit 1
