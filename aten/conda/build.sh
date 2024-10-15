#!/bin/bash

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

# When conda-build constructs a new working copy to perform a build
# in, it recursively copies *all* files and directories in the original
# source directory, including any pre-existing build products (e.g.,
# if you previously ran cmake.)  This is problematic, because if
# a 'build' directory already exists, cmake will reuse build settings
# rather than recompute them from scratch.  We want a fresh build, so
# we prophylactically remove the build directory.
rm -rf build || true

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release $CONDA_CMAKE_ARGS ..
make install -j20
