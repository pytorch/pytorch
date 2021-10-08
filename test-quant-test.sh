#!/bin/bash
PYTORCH_JIT_LOG_LEVEL=">>kernel:>>eval" \
PYTORCH_TENSOREXPR_DONT_USE_LLVM=1 \
USE_PYTORCH_QNNPACK=1 \
USE_QNNPACK=1 \
USE_FBGEMM=0 \
DEBUG=1 CXX_FLAGS="-g"  USE_LLVM=/home/ivankobzarev/llvm90install USE_XNNPACK=1 ./scripts/build_local.sh \
  -DBUILD_BINARY=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TEST=ON \
  -DUSE_LLVM=/home/ivankobzarev/llvm90install \
  && ./build/bin/test_tensorexpr --gtest_filter="Quantization.QuantAdd*"
