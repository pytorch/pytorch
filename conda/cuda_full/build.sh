#!/bin/bash

# This needs to run on Ubuntu 16.04 with gcc 5
# Install script for Anaconda environments with CUDA on linux
# This script is not supposed to be called directly, but should be run by:
#
# $ cd <path to caffe2, e.g. ~/caffe2>
# $ conda build conda/cuda_full
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

# Build with a big suite of libraries
CMAKE_ARGS=()
CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DCUDA_ARCH_NAME=All")
CMAKE_ARGS+=("-DUSE_GFLAGS=ON")
CMAKE_ARGS+=("-DUSE_GLOG=ON")
CMAKE_ARGS+=("-DUSE_GLOO=ON")
CMAKE_ARGS+=("-DUSE_IBVERBS=ON")
CMAKE_ARGS+=("-DUSE_LMDB=ON")
CMAKE_ARGS+=("-DUSE_MPI=ON")
CMAKE_ARGS+=("-DUSE_NCCL=ON")
CMAKE_ARGS+=("-DUSE_OPENCV=ON")

# cudnn comes from a module location
#-DCUDNN_ROOT_DIR=/public/apps/cudnn/v7.0/cuda/ \

# Explicitly turn unused packages off to prevent cmake from trying to find
# system libraries. If conda packages are built with any system libraries then
# they will not be relocatable.
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DUSE_REDIS=OFF")
CMAKE_ARGS+=("-DUSE_ROCKSDB=OFF")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

# No rpaths will work for anaconda?
# -DCMAKE_SKIP_RPATH=ON \


mkdir -p build
cd build
cmake "${CMAKE_ARGS[@]}"  $CONDA_CMAKE_ARGS $PYTHON_ARGS ..
make VERBOSE=1 "-j$(nproc)"

make install/fast
