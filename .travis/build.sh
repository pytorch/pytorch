#!/bin/bash

set -e

mkdir -p build
cd build

# Parallelize build with number of processors
JOBS=$(grep -c ^processor /proc/cpuinfo)

cmake .. \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DUSE_REDIS=ON \
    -DUSE_IBVERBS=ON \
    -DBUILD_TEST=ON \
    -DBUILD_BENCHMARK=ON

make -j${JOBS}
