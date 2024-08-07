#!/bin/bash

set -ex

[ -n "$NINJA_VERSION" ]

url="https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"

pushd /tmp
wget --no-verbose --output-document=ninja-linux.zip "$url"
unzip ninja-linux.zip -d /usr/local/bin
rm -f ninja-linux.zip
popd
