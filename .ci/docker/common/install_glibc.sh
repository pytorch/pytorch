#!/bin/bash

set -ex

[ -n "$GLIBC_VERSION" ]
if [[ -n "$CENTOS_VERSION" ]]; then
  [ -n "$DEVTOOLSET_VERSION" ]
fi

yum install -y wget sed

mkdir -p /packages && cd /packages
wget -q http://ftp.gnu.org/gnu/glibc/glibc-$GLIBC_VERSION.tar.gz
tar xzf glibc-$GLIBC_VERSION.tar.gz
if [[ "$GLIBC_VERSION" == "2.26" ]]; then
  cd glibc-$GLIBC_VERSION
  sed -i 's/$name ne "nss_test1"/$name ne "nss_test1" \&\& $name ne "nss_test2"/' scripts/test-installation.pl
  cd ..
fi
mkdir -p glibc-$GLIBC_VERSION-build && cd glibc-$GLIBC_VERSION-build

if [[ -n "$CENTOS_VERSION" ]]; then
  export PATH=/opt/rh/devtoolset-$DEVTOOLSET_VERSION/root/usr/bin:$PATH
fi

../glibc-$GLIBC_VERSION/configure --prefix=/usr CFLAGS='-Wno-stringop-truncation -Wno-format-overflow -Wno-restrict -Wno-format-truncation -g -O2'
make -j$(nproc)
make install

# Cleanup
rm -rf /packages
rm -rf /var/cache/yum/*
rm -rf /var/lib/rpm/__db.*
yum clean all
