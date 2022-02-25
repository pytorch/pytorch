#!/bin/bash
###############################################################################
# This script tests the flow to build libtorch locally with optimized binary
# size for mobile devices and the flow to integrate it with a simple predictor
# in c++.
#
# Supported custom build types:
#
# 1. `TEST_DEFAULT_BUILD=1 ./build.sh` - it is similar to the prebuilt libtorch
# libraries released for Android and iOS (same CMake build options + host
# toolchain), which doesn't contain autograd function nor backward ops thus is
# smaller than full LibTorch.
#
# 2. `TEST_CUSTOM_BUILD_STATIC=1 ./build.sh` - optimizes libtorch size by only
# including ops used by a specific model. It relies on the static dispatch +
# linker to prune code.
#
###############################################################################

set -ex -o pipefail

SRC_ROOT="$( cd "$(dirname "$0")"/../../.. ; pwd -P)"
TEST_SRC_ROOT="${SRC_ROOT}/test/mobile/custom_build"
BUILD_ROOT="${BUILD_ROOT:-${SRC_ROOT}/build_test_custom_build}"
mkdir -p "${BUILD_ROOT}"
cd "${BUILD_ROOT}"

prepare_model_and_dump_root_ops() {
  cd "${BUILD_ROOT}"
  MODEL="${BUILD_ROOT}/MobileNetV2.pt"
  ROOT_OPS="${BUILD_ROOT}/MobileNetV2.yaml"

  python "${TEST_SRC_ROOT}/prepare_model.py"
}

run_default_build() {
  LIBTORCH_BUILD_ROOT="${BUILD_ROOT}/build_default_libtorch"
  LIBTORCH_INSTALL_PREFIX="${LIBTORCH_BUILD_ROOT}/install"

  BUILD_ROOT="${LIBTORCH_BUILD_ROOT}" \
    "${SRC_ROOT}/scripts/build_mobile.sh"
}

run_custom_build_with_static_dispatch() {
  LIBTORCH_BUILD_ROOT="${BUILD_ROOT}/build_custom_libtorch_static"
  LIBTORCH_INSTALL_PREFIX="${LIBTORCH_BUILD_ROOT}/install"

  # Here it generates registration code for used ROOT ops only, whose unboxing
  # kernels are still needed by the JIT runtime. The intermediate ops will be
  # automatically kepted by the linker as they are statically referenced by the
  # static dispatch code, for which we can bypass the registration.
  # We don't set '-DSTATIC_DISPATCH_BACKEND=CPU' explicitly to test automatic
  # fallback to static dispatch.
  BUILD_ROOT="${LIBTORCH_BUILD_ROOT}" \
    "${SRC_ROOT}/scripts/build_mobile.sh" \
    -DCMAKE_CXX_FLAGS="-DSTRIP_ERROR_MESSAGES" \
    -DSELECTED_OP_LIST="${ROOT_OPS}"
}

build_predictor() {
  PREDICTOR_BUILD_ROOT="${BUILD_ROOT}/predictor"

  rm -rf "${PREDICTOR_BUILD_ROOT}" && mkdir -p "${PREDICTOR_BUILD_ROOT}"
  cd "${PREDICTOR_BUILD_ROOT}"

  cmake "${TEST_SRC_ROOT}" \
    -DCMAKE_PREFIX_PATH="${LIBTORCH_INSTALL_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release

  make
}

run_predictor() {
  cd "${PREDICTOR_BUILD_ROOT}"
  ./Predictor "${MODEL}" > output.txt

  if cmp -s output.txt "${TEST_SRC_ROOT}/expected_output.txt"; then
    echo "Test result is the same as expected."
  else
    echo "Test result is DIFFERENT from expected!"
    diff output.txt "${TEST_SRC_ROOT}/expected_output.txt"
    exit 1
  fi
}

test_default_build() {
  prepare_model_and_dump_root_ops
  run_default_build
  build_predictor
  run_predictor
}

test_custom_build_with_static_dispatch() {
  prepare_model_and_dump_root_ops
  run_custom_build_with_static_dispatch
  build_predictor
  run_predictor
}

if [ -n "${TEST_DEFAULT_BUILD}" ]; then
  test_default_build
fi

if [ -n "${TEST_CUSTOM_BUILD_STATIC}" ]; then
  test_custom_build_with_static_dispatch
fi
