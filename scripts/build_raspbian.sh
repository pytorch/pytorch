#!/bin/bash
##############################################################################
# Example command to build the Raspbian target.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for raspbian. The build
# is essentially much similar to a host build, with one additional change
# which is to specify -mfpu=neon for optimized speed.

CAFFE2_ROOT="$( cd "$(dirname -- "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}
mkdir -p $BUILD_ROOT
echo "Build Caffe2 raspbian into: $BUILD_ROOT"

# obtain dependencies.
echo "Installing dependencies."
sudo apt-get install \
  cmake \
  libgflags-dev \
  libgoogle-glog-dev \
  libprotobuf-dev \
  libpython-dev \
  python-pip \
  python-numpy \
  protobuf-compiler \
  python-protobuf
# python dependencies
sudo pip install hypothesis

# Now, actually build the raspbian target.
echo "Building caffe2"
cd $BUILD_ROOT

# Note: you can add more dependencies above if you need libraries such as
# leveldb, lmdb, etc.
cmake "$CAFFE2_ROOT" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCAFFE2_CPU_FLAGS="-mfpu=neon -mfloat-abi=hard" \
    || exit 1

# Note: while Raspberry pi has 4 cores, running too many builds in parallel may
# cause out of memory errors so we will simply run -j 2 only.
make -j 2 || exit 1
