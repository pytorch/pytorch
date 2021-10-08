#!/bin/bash
DEBUG=1 CXX_FLAGS="-g"  USE_LLVM=/home/ivankobzarev/llvm90install ./scripts/build_local.sh \
  -DBUILD_BINARY=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TEST=ON \
  -DUSE_LLVM=/home/ivankobzarev/llvm90install && sh ./aot-quantdequant.sh
