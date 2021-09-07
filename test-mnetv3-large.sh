DEBUG=1 USE_LLVM=/home/ivankobzarev/llvm90install USE_XNNPACK=1 ./scripts/build_local.sh \
  -DBUILD_BINARY=ON \
  -DBUILD_TEST=ON \
  -DUSE_LLVM=/home/ivankobzarev/llvm90install && sh ./aot-mnetv3-large.sh
