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

print_cmake_info() {
  CMAKE_EXEC=$(which cmake)
  echo "$CMAKE_EXEC"

  CONDA_INSTALLATION_DIR=$(dirname "$CMAKE_EXEC")
  # Print all libraries under cmake rpath for debugging
  ls -la "$CONDA_INSTALLATION_DIR/../lib"

  export CMAKE_EXEC
  # Explicitly add conda env lib folder to cmake rpath to address the flaky issue
  # where cmake dependencies couldn't be found. This seems to point to how conda
  # links $CMAKE_EXEC to its package cache when cloning a new environment
  install_name_tool -add_rpath @executable_path/../lib "${CMAKE_EXEC}" || true
  # Adding the rpath will invalidate cmake signature, so signing it again here
  # to trust the executable. EXC_BAD_ACCESS (SIGKILL (Code Signature Invalid))
  # with an exit code 137 otherwise
  codesign -f -s - "${CMAKE_EXEC}" || true
}
