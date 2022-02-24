# This script should be called from .jenkins/pytorch/build.sh. Assuming we are at pytorch source root directory.

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
echo "Build lite interpreter with lightweight dispatch."

# prepare test
TEST_SRC_ROOT="$PWD/test/mobile/lightweight_dispatch"
python $TEST_SRC_ROOT/tests_setup.py setup

CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-${PWD}/../}
mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

LIGHTWEIGHT_DISPATCH_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/lightweight-dispatch-build"
BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py

mkdir -p "$LIGHTWEIGHT_DISPATCH_BUILD"
pushd "$LIGHTWEIGHT_DISPATCH_BUILD"

export USE_LIGHTWEIGHT_DISPATCH=1
export STATIC_DISPATCH_BACKEND="CPU"
export BUILD_LITE_INTERPRETER=1

VERBOSE=1 DEBUG=1 python "${BUILD_LIBTORCH_PY}"

popd
