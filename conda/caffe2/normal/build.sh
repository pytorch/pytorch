#!/bin/bash

# Install script for Anaconda environments on macOS and linux.
# This script is not supposed to be called directly, but should be called by
# scripts/build_anaconda.sh, which handles setting lots of needed flags
# depending on the current system and user flags.
#
# If you're debugging this, it may be useful to use the env that conda build is
# using:
# $ cd <anaconda_root>/conda-bld/caffe2_<timestamp>
# $ source activate _h_env_... # some long path with lots of placeholders
#
# Also, failed builds will accumulate those caffe2_<timestamp> directories. You
# can remove them after a succesfull build with
# $ conda build purge

set -ex

echo "Installing caffe2 to ${PREFIX}"

# Install under specified prefix
cmake_args=()
cmake_args+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
cmake_args+=("-DCMAKE_PREFIX_PATH=$PREFIX")

# Build Caffe2
mkdir -p build
cd build
cmake "${cmake_args[@]}" $CAFFE2_CMAKE_ARGS ..
if [ "$(uname)" == 'Darwin' ]; then
  make "-j$(sysctl -n hw.ncpu)"
else
  make "-j$(nproc)"
fi

make install/fast
