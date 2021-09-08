#!/bin/bash

set -ex

git clone https://github.com/malfet/breakpad.git -b pytorch/release-1.9
pushd breakpad

git clone https://chromium.googlesource.com/linux-syscall-support src/third_party/lss
pushd src/third_party/lss
# same as with breakpad, there are no real releases for this repo so use a
# commit as the pin
git checkout e1e7b0ad8ee99a875b272c8e33e308472e897660
popd

./configure
make
make install
popd
rm -rf breakpad
