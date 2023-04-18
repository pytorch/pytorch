#!/bin/bash
set -euxo pipefail

rm -rf build
mkdir -p build
cd build
TORCH_PATH=$PWD/../../../../../torch
cmake .. -DCMAKE_PREFIX_PATH=$TORCH_PATH
make
LD_LIBRARY_PATH=$TORCH_PATH/lib ./test
