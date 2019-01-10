#!/bin/bash

set -ex

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

# Install sccache from binary release.
# Note: this release does NOT yet work with nvcc.
pushd /tmp
curl -LOs https://github.com/mozilla/sccache/releases/download/0.2.5/sccache-0.2.5-x86_64-unknown-linux-musl.tar.gz
tar -zxvf sccache-0.2.5-x86_64-unknown-linux-musl.tar.gz
mv sccache-0.2.5-x86_64-unknown-linux-musl/sccache /usr/local/bin/sccache
rm -rf sccache-0.2.5-x86_64-unknown-linux-musl*
popd
