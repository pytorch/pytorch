#!/bin/bash

set -ex

[ -n "$GCC_VERSION" ]

apt-get update
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y --no-install-recommends gcc-"$GCC_VERSION" g++-"$GCC_VERSION"
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use update-alternatives to make this version the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"$GCC_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"$GCC_VERSION" 50
