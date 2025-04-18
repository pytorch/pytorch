#!/bin/bash

set -ex

[ -n "$NINJA_VERSION" ]

arch=$(uname -m)
if [ "$arch" == "aarch64" ]; then
    url="https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux-aarch64.zip"
else
    url="https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"
fi

pushd /tmp
wget --no-verbose --output-document=ninja-linux.zip "$url"
unzip ninja-linux.zip -d /usr/local/bin
rm -f ninja-linux.zip
popd