#!/bin/bash

set -ex

[ -n "$CLANG_VERSION" ]
[ -n "$UBUNTU_VERSION" ]

if [[ "$CLANG_VERSION" == "6.0" || "$CLANG_VERSION" == "7" || "$CLANG_VERSION" == "8" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends software-properties-common wget
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
  if [[ "$UBUNTU_VERSION" == 16.04 ]]; then
      apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-${CLANG_VERSION} main"
  elif [[ "$UBUNTU_VERSION" == 17.10 ]]; then
      apt-add-repository "deb http://apt.llvm.org/artful/ llvm-toolchain-artful-${CLANG_VERSION} main"
  elif [[ "$UBUNTU_VERSION" == 18.04 ]]; then
      apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-${CLANG_VERSION} main"
  elif [[ "$UBUNTU_VERSION" == 18.10 ]]; then
      apt-add-repository "deb http://apt.llvm.org/cosmic/ llvm-toolchain-cosmic-${CLANG_VERSION} main"
  else
      echo "Invalid Ubuntu version: ${UBUNTU_VERSION}"
      exit 1
  fi
fi

apt-get update
apt-get install -y --no-install-recommends clang-"$CLANG_VERSION" libclang-"$CLANG_VERSION"-dev
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use update-alternatives to make this version the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-"$CLANG_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-"$CLANG_VERSION" 50
