#!/bin/bash

set -ex

git clone https://github.com/google/breakpad.git
cd breakpad

git clone https://chromium.googlesource.com/linux-syscall-support src/third_party/lss
./configure
make
make install
cd ..
rm -rf breakpad
