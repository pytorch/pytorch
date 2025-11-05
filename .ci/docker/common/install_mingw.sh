#!/bin/bash

set -ex

# Install MinGW-w64 for Windows cross-compilation
apt-get update
apt-get install -y g++-mingw-w64-x86-64-posix

echo "MinGW-w64 installed successfully"
x86_64-w64-mingw32-g++ --version
