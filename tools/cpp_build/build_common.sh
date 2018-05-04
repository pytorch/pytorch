#!/usr/bin/env bash

BUILD_PATH="${1:-$SCRIPTPATH/build}"
INSTALL_PREFIX="$BUILD_PATH/install"
PYTORCHPATH="$SCRIPTPATH/../.."

NO_CUDA=ON
if [ -x "$(command -v nvcc)" ]; then
  NO_CUDA=OFF
fi

ATEN_BUILDPATH="$BUILD_PATH/aten"
NANOPB_BUILDPATH="$BUILD_PATH/nanopb"
LIBTORCH_BUILDPATH="$BUILD_PATH/libtorch"

# Build with Ninja if available. It has much cleaner output.
GENERATE="Unix Makefiles"
MAKE=make
if [ -x "$(command -v ninja)" ]; then
  GENERATE=Ninja
  MAKE=ninja
fi

# Code is developed a lot more than released, so default to Debug.
BUILD_TYPE=${BUILD_TYPE:-Debug}

# Try to build with as many threads as we have cores, default to 4 if the
# command fails.
set +e
if [[ "$(uname)" == "Linux" ]]; then
  # https://stackoverflow.com/questions/6481005/how-to-obtain-the-number-of-cpus-cores-in-linux-from-the-command-line
  JOBS="$(grep -c '^processor' /proc/cpuinfo)"
else # if [[ "$(uname)" == "Darwin"]]
  # https://stackoverflow.com/questions/1715580/how-to-discover-number-of-logical-cores-on-mac-os-x
  JOBS="$(sysctl -n hw.ncpu)"
fi
set -e
if [[ $? -ne 0 ]]; then
  JOBS=4
fi
