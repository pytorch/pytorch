#!/bin/bash

set -ex

[ -n "$CLANG_VERSION" ]

apt-get update
apt-get install -y --no-install-recommends clang-"$CLANG_VERSION"
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use update-alternatives to make this version the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-"$CLANG_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-"$CLANG_VERSION" 50
