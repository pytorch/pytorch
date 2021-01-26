#!/bin/bash

set -ex

git clone https://github.com/NixOS/patchelf
cd patchelf
sed -i 's/serial/parallel/g' configure.ac
./bootstrap.sh
./configure
make
make install
cd ..
rm -rf patchelf
