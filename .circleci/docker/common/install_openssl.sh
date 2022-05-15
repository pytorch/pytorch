#!/bin/bash

set -ex

OPENSSL=openssl-1.1.1k

wget -q -O "${OPENSSL}.tar.gz" "https://ossci-linux.s3.amazonaws.com/${OPENSSL}.tar.gz"
tar xf "${OPENSSL}.tar.gz"
cd "${OPENSSL}"
./config --prefix=/opt/openssl -d '-Wl,--enable-new-dtags,-rpath,$(LIBRPATH)'
# NOTE: openssl install errors out when built with the -j option
make -j6; make install_sw
cd ..
rm -rf "${OPENSSL}"
