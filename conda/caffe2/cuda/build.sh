#!/bin/bash

# Install script for Anaconda environments with CUDA on linux
# This script is not supposed to be called directly, but should be run by:
#
# $ cd <path to caffe2, e.g. ~/caffe2>
# $ conda build conda/build
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

# This is needed for build variants (packages with multiple variants in 
# conda_build_config.yaml) to remove any files that cmake cached, since
# conda-build uses the same environment for all the build variants
rm -rf build

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()

# Build with minimal required libraries
# Add CMAKE flags here
CMAKE_ARGS+=("-DUSE_MPI=OFF")

# Build with CUDA
CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DUSE_NCCL=ON")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

mkdir -p build
cd build
cmake "${CMAKE_ARGS[@]}"  $CONDA_CMAKE_BUILD_ARGS $PYTHON_ARGS ..
make "-j$(nproc)"

make install/fast
