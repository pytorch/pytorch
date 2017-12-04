#!/bin/bash

set -ex

apt-get update
apt-get install -y --no-install-recommends autoconf
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install ccache from source.
# Needs specific branch to work with nvcc (ccache/ccache#145)
# Also pulls in a commit that disables documentation generation,
# as this requires asciidoc to be installed (which pulls in a LOT of deps).
pushd /tmp
git clone https://github.com/pietern/ccache -b ccbin
pushd ccache
./autogen.sh
./configure --prefix=/usr/local
make "-j$(nproc)" install
popd
popd
