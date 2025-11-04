#!/bin/bash
# This script should be called from .ci/pytorch/build.sh. Assuming we are at pytorch source root directory.

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

set -ex -o pipefail

# shellcheck disable=SC2034
echo "Build lite interpreter with lightweight dispatch."

CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-"build/custom_test_artifacts"}
mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

BUILD_LIBTORCH_PY="$PWD/tools/build_libtorch.py"
TEST_SRC_ROOT="$PWD/test/mobile/lightweight_dispatch"

pushd "$CUSTOM_TEST_ARTIFACT_BUILD_DIR"

# prepare test
OP_LIST="lightweight_dispatch_ops.yaml"
export SELECTED_OP_LIST=$TEST_SRC_ROOT/$OP_LIST
python "$TEST_SRC_ROOT/tests_setup.py" setup "$SELECTED_OP_LIST"

export USE_DISTRIBUTED=0
export USE_LIGHTWEIGHT_DISPATCH=1
export STATIC_DISPATCH_BACKEND="CPU"
export BUILD_LITE_INTERPRETER=1

export USE_FBGEMM=0
python "${BUILD_LIBTORCH_PY}"
ret=$?

if [ "$ret" -ne 0 ]; then
  echo "Lite interpreter build failed!"
  exit "$ret"
fi


# run test
if ! build/bin/test_codegen_unboxing; then
  echo "test_codegen_unboxing has failure!"
  exit 1
fi

# shutdown test
python "$TEST_SRC_ROOT/tests_setup.py" shutdown "$SELECTED_OP_LIST"

popd

exit 0
