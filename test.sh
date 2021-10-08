#!/bin/bash
DEBUG=1 CXX_FLAGS="-g"  USE_LLVM=/home/ivankobzarev/llvm90install USE_XNNPACK=1 ./scripts/build_local.sh \
  -DBUILD_BINARY=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TEST=ON \
  -DUSE_LLVM=/home/ivankobzarev/llvm90install && ./build/bin/test_tensorexpr --gtest_filter="Kernel.Linear"
