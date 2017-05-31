#!/bin/bash
set -e
set -x

export CXX=/usr/lib/ccache/g++

mkdir build
cd build
cmake .. \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DUSE_CUDA=OFF -DUSE_NCCL=OFF -DUSE_GLOO=OFF \
    -DUSE_NNPACK=OFF
make -j$(nproc)
