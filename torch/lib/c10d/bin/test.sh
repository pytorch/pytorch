#!/bin/bash

set -ex

mkdir -p build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX="$PWD/../"
make all test
