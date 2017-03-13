#!/bin/bash

mkdir build
cd build

cmake .. \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DUSE_REDIS=ON \
    -DUSE_IBVERBS=ON \
    && make