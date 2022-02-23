# This script should be called from .jenkins/pytorch/build.sh. Assuming we are at pytorch source root directory.

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"
echo "Build lite interpreter with lightweight dispatch."
USE_DISTRIBUTED=0 \
  USE_MKLDNN=0 \
  USE_FBGEMM=0 \
  USE_NNPACK=0 \
  USE_QNNPACK=0 \
  USE_XNNPACK=0 \
  USE_LIGHTWEIGHT_DISPATCH=1 \
  STATIC_DISPATCH_BACKEND="CPU" \
  BUILD_LITE_INTERPRETER=1 \
  BUILD_TEST=1 \
  INSTALL_TEST=1 \
  BUILD_MOBILE_TEST=1 \
  python setup.py bdist_wheel
  python -mpip install dist/*.whl

CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-${PWD}/../}
mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

LIGHTWEIGHT_DISPATCH_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/lightweight-dispatch-build"
TEST_SRC_ROOT="$PWD/test/mobile/lightweight_dispatch"
SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"

mkdir -p "$LIGHTWEIGHT_DISPATCH_BUILD"
pushd "$LIGHTWEIGHT_DISPATCH_BUILD"
cmake "$TEST_SRC_ROOT" -DCMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_codegen_unboxing
make VERBOSE=1
popd

assert_git_not_dirty