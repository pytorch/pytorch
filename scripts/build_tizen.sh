#!/usr/bin/env bash
##############################################################################
#  Example command to build the Tizen target (RPi3).
##############################################################################
#
# This script shows how one can build a Caffe2 binary for a Tizen device (RPi3).
# The build is essentially much similar to a host build, with one additional change
# which is to specify -mfpu=neon for optimized speed.

setup_environment(){
# The rootfs image for a Tizen target (RPi3)is located at the below webpage:
# http://download.tizen.org/releases/milestone/tizen/4.0.m1/tizen-unified_20170529.1/images/
# If you do not have a Tizen device, Please, run qemu-arm-static and chroot command.
# $ sudo chroot ~/tizen-rootfs qemu-arm-static /usr/bin/bash

CAFFE2_ROOT="$( cd "$(dirname -- "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build"}
mkdir -p $BUILD_ROOT
echo "Build Caffe2 Tizen into: $BUILD_ROOT"
}

caffe2_lite_dep_packages(){
# Obtain necessary dependencies
# You can set-up a rpm repository with zypper, yum, and dnf because Tizen
# software platform officially support rpm format such as Fedora, OpenSUSE.
# The official Tizen repository is as following:
# http://download.tizen.org/releases/milestone/tizen/4.0.m1/
echo "Installing dependencies."
sudo zypper install \
  make \
  strace \
  cmake \
  gcc* \
  binutils \
  glibc* \
  cpp \
  protobuf-devel \
  libstdc++*
}

caffe2_lite_build(){
# Now, actually build the android target.
echo "Building caffe2"
cd $BUILD_ROOT

# Note: add more dependencies above if you need libraries such as leveldb, lmdb, etc.
# If you have to disable a specific package due to a package absence
# from https://git.tizen.org/cgit/, append -Dxxx_xxx=OFF option before executing cmake.
cmake .. \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DUSE_GFLAGS=OFF  \
    -DUSE_GLOG=OFF -DUSE_NNPACK=OFF \
    -DRUN_HAVE_STD_REGEX=0 \
    -DRUN_HAVE_POSIX_REGEX=0 \
    -DHAVE_GNU_POSIX_REGEX=0 \
    -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=OFF \
    -DBUILD_PYTHON=OFF \
    -DUSE_GLOO=OFF \
    -DUSE_OPENCV=OFF \
    -DCAFFE2_CPU_FLAGS="-mfpu=neon -mfloat-abi=soft" \
    || exit 1

make -j`nproc` || exit 1
}

caffe2_full_dep_packages(){
# Obtain necessary dependencies
# You can set-up a rpm repository with zypper, yum, and dnf because Tizen
# software platform officially support rpm format such as Fedora, OpenSUSE.
# The official Tizen repository is as following:
# http://download.tizen.org/releases/milestone/tizen/4.0.m1/
echo "Installing dependencies."
sudo zypper install \
  cmake \
  libgflags-dev \
  libgoogle-glog-dev \
  libprotobuf-dev \
  protobuf-compiler

# Obtain optional dependencies that are usually useful to have.
echo "Installing optional dependencies."
sudo zypper install \
  libleveldb-dev \
  liblmdb-dev \
  libpython-dev \
  libsnappy-dev \
  python-numpy \
  python-pip \
  python-protobuf

# Obtain python hypothesis, which Caffe2 uses for unit testing. Note that
# the one provided by zypper is quite old so we install it via pip
sudo pip install hypothesis
}

caffe2_full_build(){
# Now, actually build the android target.
echo "Building caffe2"
cd $BUILD_ROOT

# Note: add more dependencies above if you need libraries such as leveldb, lmdb, etc.
# If you have to disable a specific package due to a package absence
# from https://git.tizen.org/cgit/, append -Dxxx_xxx=OFF option before executing cmake.
cmake "$CAFFE2_ROOT" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DUSE_CUDA=OFF \
    -DUSE_ITT=OFF \
    -DUSE_OPENCV=OFF \
    -DUSE_LMDB=OFF \
    -DCAFFE2_CPU_FLAGS="-mfpu=neon -mfloat-abi=soft" \
    || exit 1

make -j`nproc` || exit 1
}

#### Main
# Setup a build environment to compile Caffe2 deeplearning framework in Tizen platform.
setup_environment
# There are two build options to support 'full' version and 'lite' version (by default).
caffe2_lite_dep_packages
caffe2_lite_build
