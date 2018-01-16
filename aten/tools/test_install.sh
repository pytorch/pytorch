#!/bin/bash
set -x
set -e
INSTALL_ROOT=$1
SRC_ROOT=$2
mkdir test_build
cd test_build
cmake -DCMAKE_PREFIX_PATH=$INSTALL_ROOT $SRC_ROOT/src/ATen/test/test_install
make
./main
