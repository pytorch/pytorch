#!/bin/bash

set -e

mkdir -p build
cd build

# Parallelize build with number of processors
JOBS=$(grep -c ^processor /proc/cpuinfo)

# Setup CMake flags based on environment variables
CMAKE_ARGS="-DUSE_REDIS=ON -DUSE_IBVERBS=ON"
if [ "${BUILD_CUDA}" == "ON" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON"
fi

cmake .. \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DBUILD_TEST=ON \
    -DBUILD_BENCHMARK=ON \
    ${CMAKE_ARGS}

# Ignore # of processors for now; perform sequential build
make -j1
