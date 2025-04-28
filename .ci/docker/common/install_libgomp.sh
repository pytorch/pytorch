#!/bin/bash
# Script used only in CD pipeline

set -ex

source /opt/rh/gcc-toolset-11/enable

# install dependencies
dnf -y install gmp-devel libmpc-devel texinfo

cd /usr/local/src
# fetch source for gcc 11
curl -LO https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.xz
tar xf gcc-11.4.0.tar.xz

mkdir -p gcc-11.4.0/build-gomp
cd gcc-11.4.0/build-gomp

# configure gcc build
CFLAGS="-O2 -march=armv8-a -mtune=generic" \
CXXFLAGS="-O2 -march=armv8-a -mtune=generic" \
LDFLAGS="-Wl,--as-needed" \
../configure --prefix=/usr --libdir=/usr/lib64 --enable-languages=c,c++ --disable-multilib --disable-bootstrap --enable-libgomp

# only build libgomp
make -j$(nproc) all-target-libgomp

make install-target-libgomp