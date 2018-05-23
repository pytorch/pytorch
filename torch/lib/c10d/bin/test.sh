#!/bin/bash

set -ex

mkdir -p build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX="$PWD/../../tmp_install"
make all test
