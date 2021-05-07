#!/bin/bash

set -ex

git clone https://github.com/driazati/breakpad.git
pushd breakpad

# breakpad has no actual releases, so this is pinned to the top commit from
# main when this was forked (including the one patch commit)
git checkout 5485e473ed46d065e05489e50dfc59d90dfd7e22

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
