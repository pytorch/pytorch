#!/bin/bash
# Script used only in CD pipeline

set -ex

LIBPNG_VERSION=1.6.37

mkdir -p libpng
pushd libpng

wget http://download.sourceforge.net/libpng/libpng-$LIBPNG_VERSION.tar.gz
tar -xvzf libpng-$LIBPNG_VERSION.tar.gz

pushd libpng-$LIBPNG_VERSION

./configure
make
make install

popd

popd
rm -rf libpng
