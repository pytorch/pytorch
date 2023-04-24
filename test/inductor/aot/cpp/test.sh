#!/bin/bash
set -euxo pipefail

rm -rf build
mkdir -p build
cd build
cmake ..
make
./test
