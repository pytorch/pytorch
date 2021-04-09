#!/bin/bash

set -ex

# Get code
git clone https://github.com/google/breakpad.git
cd breakpad
# Checkout a commit from the head of the Chrome 90 branch
git checkout b324760c7f53667af128a6b77b790323da04fcb9

# Get dependency for Linux builds
git clone https://chromium.googlesource.com/linux-syscall-support src/third_party/lss
cd src/third_party/lss
# Pin to a commit that was working when this was committed
git checkout 29f7c7e018f4ce706a709f0b0afbf8bacf869480
cd ../../..

# Build
./configure
make
make install

# Cleanup
cd ..
rm -rf breakpad
