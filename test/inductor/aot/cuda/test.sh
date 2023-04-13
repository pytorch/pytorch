#!/bin/bash
set -euxo pipefail

rm -rf build
mkdir -p build
cd build
cmake ..
make
LD_LIBRARY_PATH=../../../../../torch/lib/ ./test
