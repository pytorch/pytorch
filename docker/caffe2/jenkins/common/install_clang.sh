#!/bin/bash

set -ex

[ -n "$CLANG_VERSION" ]

if [[ "$CLANG_VERSION" == "7" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends software-properties-common wget
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
  apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main"
fi

apt-get update
apt-get install -y --no-install-recommends clang-"$CLANG_VERSION"
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use update-alternatives to make this version the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-"$CLANG_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-"$CLANG_VERSION" 50
