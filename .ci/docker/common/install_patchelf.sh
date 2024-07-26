#!/bin/bash
# Script used only in CD pipeline

set -ex

# Pin the version to latest release 0.17.2, building newer commit starts
# to fail on the current image
git clone -b 0.17.2 --single-branch https://github.com/NixOS/patchelf
cd patchelf
sed -i 's/serial/parallel/g' configure.ac
./bootstrap.sh
./configure
make
make install
cd ..
rm -rf patchelf
