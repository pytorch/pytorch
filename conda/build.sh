#!/bin/bash

# Install script for Anaconda environments on macOS. This will only work on mac
# This script is not supposed to be called correctly, but should be run by:
#
# $ cd <path to caffe2, e.g. ~/caffe2>
# $ conda build conda/build
#
# This installation uses MKL and does not use CUDA
#
# If you're debugging this, it may be useful to use the env that conda build is
# using:
# $ cd <anaconda_root>/conda-bld/caffe2_<timestamp>
# $ source activate _h_env_... # some long path with lots of placeholders
#
# Also, failed builds will accumulate those caffe2_<timestamp> directories. You
# can remove them after a succesfull build with
# $ conda build purge
#

set -ex

echo "Installing caffe2 to ${PREFIX}"

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()

# Default leveldb from conda-forge doesn't work. If you want to use leveldb,
# use this old pip version
# pip install leveldb==0.18
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")

# This installation defaults to using MKL because it is much faster. If you
# want to build without MKL then you should also remove mkl from meta.yaml in
# addition to removing the flags below
CMAKE_ARGS+=("-DBLAS=MKL")
CMAKE_ARGS+=("-DMKL_INCLUDE_DIR=$CONDA_PREFIX/include")

# The following libraries are incompatible with macOS builds
CMAKE_ARGS+=("-DUSE_GLOO=OFF")
CMAKE_ARGS+=("-DUSE_MOBILE_OPENGL=OFF")
CMAKE_ARGS+=("-DUSE_METAL=OFF")

# It is rare for macs to have a GPU card with CUDA support, but if you have one
# then remove these flags
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_NCCL=OFF")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

mkdir -p build
cd build
cmake "${CMAKE_ARGS[@]}"  $CONDA_CMAKE_ARGS $PYTHON_ARGS ..
if [ "$(uname)" == 'Darwin' ]; then
  make "-j$(sysctl -n hw.ncpu)"
else
  make "-j$(nproc)"
fi

make install/fast

# Python libraries got installed to wrong place, so move them
# to the right place. See https://github.com/caffe2/caffe2/issues/1015
echo "Installing Python to $SP_DIR"
mv $PREFIX/caffe2 $SP_DIR
