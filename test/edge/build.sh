#!/bin/bash
# This script should be called from .jenkins/pytorch/build.sh. Assuming we are at pytorch source root directory.
# It builds Executorch test binary and runs it.
# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

set -ex -o pipefail

# shellcheck disable=SC2034
echo "Building Executorch."

CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-"build/custom_test_artifacts"}
mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

BUILD_LIBTORCH_PY="$PWD/tools/build_libtorch.py"
TEST_SRC_ROOT="$PWD/test/edge"

cmake . -B "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/build"
pushd "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

export USE_DISTRIBUTED=0
export USE_FBGEMM=0
export BUILD_EXECUTORCH=1
export INSTALL_TEST=1
export BUILD_TEST=1
# Need to build libtorch because we are testing Executorch in ATen mode.
python "${BUILD_LIBTORCH_PY}"

#cmake "${TEST_SRC_ROOT}"

ret=$?

if [ "$ret" -ne 0 ]; then
  echo "Libtorch build failed!"
  exit "$ret"
fi

# run test
if ! build/bin/test_edge_op_registration; then
  echo "test_edge_op_registration has failure!"
  exit 1
fi


popd

exit 0