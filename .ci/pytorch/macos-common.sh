#!/bin/bash

# Common prelude for macos-build.sh and macos-test.sh

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sysctl -a | grep machdep.cpu

# These are required for both the build job and the test job.
# In the latter to test cpp extensions.
export MACOSX_DEPLOYMENT_TARGET=11.1
export CXX=clang++
export CC=clang

brew install cmake
conda remove --yes cmake
# shellcheck disable=SC2155
export PATH="$PATH:$(brew --prefix)/bin"

print_cmake_info() {
  CMAKE_EXEC=$(which cmake)
  echo "$CMAKE_EXEC"

  CMAKE_INSTALLATION_DIR=$(dirname "$CMAKE_EXEC")
  # Print all libraries under cmake rpath for debugging
  ls -la "$CMAKE_INSTALLATION_DIR/../lib"

  export CMAKE_EXEC
}
