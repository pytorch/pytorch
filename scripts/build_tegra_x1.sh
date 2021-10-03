#!/bin/bash
##############################################################################
# Example command to build Caffe2 on Tegra X1.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for NVidia's TX1.
# The build script assumes that you have the most recent libraries installed
# via the JetPack toolkit available at
#     https://developer.nvidia.com/embedded/jetpack
# and it assumes that we are starting from a fresh system after the jetpack
# installation. If you have already installed some of the dependencies, you
# may be able to skip quite a few of the apt-get installs.

CAFFE2_ROOT="$( cd "$(dirname -- "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}
mkdir -p $BUILD_ROOT
echo "Build Caffe2 raspbian into: $BUILD_ROOT"

# obtain necessary dependencies
echo "Installing dependencies."
sudo apt-get install \
  cmake \
  libgflags-dev \
  libgoogle-glog-dev \
  libprotobuf-dev \
  protobuf-compiler

# obtain optional dependencies that are usually useful to have.
echo "Installing optional dependencies."
sudo apt-get install \
  libleveldb-dev \
  liblmdb-dev \
  libpython-dev \
  libsnappy-dev \
  python-numpy \
  python-pip \
  python-protobuf

# Obtain python hypothesis, which Caffe2 uses for unit testing. Note that
# the one provided by apt-get is quite old so we install it via pip
sudo pip install hypothesis

# Install the six module, which includes Python 2 and 3 compatibility utilities,
# and is required for Caffe2
sudo pip install six

# Now, actually build the android target.
echo "Building caffe2"
cd $BUILD_ROOT

# CUDA_USE_STATIC_CUDA_RUNTIME needs to be set to off so that opencv can be
# properly used. Otherwise, opencv will complain that opencv_dep_cudart cannot
# be found.
cmake "$CAFFE2_ROOT" -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    || exit 1

make -j 4 || exit 1
