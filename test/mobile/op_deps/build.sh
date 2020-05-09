#!/bin/bash
##############################################################################
# A simple project that uses C10 op registration API to create a bunch of
# inter-dependent dummy ops in order to test op dependency analysis script for
# mobile custom build workflow.
##############################################################################

set -ex

SRC_ROOT="$( cd "$(dirname "$0")"/../../.. ; pwd -P)"
BUILD_ROOT="${BUILD_ROOT:-${SRC_ROOT}/build_test_op_deps}"
INSTALL_PREFIX="${BUILD_ROOT}/install"

mkdir -p "${BUILD_ROOT}"
cd "${BUILD_ROOT}"

if [ ! -d "${TORCH_INSTALL_PREFIX:=${SRC_ROOT}/build_mobile/install}" ]; then
  echo "Unable to find torch library in ${TORCH_INSTALL_PREFIX}"
  exit 1
fi

cmake "${SRC_ROOT}/test/mobile/op_deps" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${TORCH_INSTALL_PREFIX}" \
  "$@" # Use-specified CMake arguments

cmake --build . --target install -- "-j${MAX_JOBS}"
echo "Installation completed: ${INSTALL_PREFIX}"
