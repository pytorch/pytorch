#!/bin/bash
###############################################################################
# This script tests the flow to build libtorch locally with optimized binary
# size for mobile devices and the flow to integrate it with a simple predictor
# in c++.
#
# There are three custom build types:
#
# 1. `TEST_DEFAULT_BUILD=1 ./build.sh` - it is similar to the prebuilt libtorch
# libraries released for Android and iOS (same CMake build options + host
# toolchain), which doesn't contain autograd function nor backward ops thus is
# smaller than full LibTorch.
#
# 2. `TEST_CUSTOM_BUILD_STATIC=1 ./build.sh` - it further optimizes libtorch
# size by only including ops used by a specific model.
#
# 3. `TEST_CUSTOM_BUILD_DYNAMIC=1 ./build.sh` - similar as 2) except that it
# relies on the op dependency graph (instead of static dispatch) to calculate
# and keep all transitively dependent ops by the model.
# Note that LLVM_DIR environment variable should be set to the location of
# LLVM-dev toolchain.
#
# Type 2) will be deprecated by type 3) in the future.
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

generate_op_dependency_graph() {
  # Regular users should get this graph from prebuilt package.
  ANALYZER_BUILD_ROOT="${BUILD_ROOT}/build_analyzer"
  OP_DEPENDENCY="${ANALYZER_BUILD_ROOT}/work/torch_result.yaml"

  if [ ! -f "${OP_DEPENDENCY}" ]; then
    BUILD_ROOT="${ANALYZER_BUILD_ROOT}" \
      ANALYZE_TORCH=1 \
      "${SRC_ROOT}/tools/code_analyzer/build.sh"
  fi
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

  BUILD_ROOT="${LIBTORCH_BUILD_ROOT}" \
    "${SRC_ROOT}/scripts/build_mobile.sh" \
    -DCMAKE_CXX_FLAGS="-DSTRIP_ERROR_MESSAGES" \
    -DUSE_STATIC_DISPATCH=ON \
    -DSELECTED_OP_LIST="${ROOT_OPS}"
}

run_custom_build_with_dynamic_dispatch() {
  LIBTORCH_BUILD_ROOT="${BUILD_ROOT}/build_custom_libtorch_dynamic"
  LIBTORCH_INSTALL_PREFIX="${LIBTORCH_BUILD_ROOT}/install"

  BUILD_ROOT="${LIBTORCH_BUILD_ROOT}" \
    "${SRC_ROOT}/scripts/build_mobile.sh" \
    -DCMAKE_CXX_FLAGS="-DSTRIP_ERROR_MESSAGES" \
    -DUSE_STATIC_DISPATCH=OFF \
    -DSELECTED_OP_LIST="${ROOT_OPS}" \
    -DOP_DEPENDENCY="${OP_DEPENDENCY}"
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

test_custom_build_with_dynamic_dispatch() {
  prepare_model_and_dump_root_ops
  generate_op_dependency_graph
  run_custom_build_with_dynamic_dispatch
  build_predictor
  run_predictor
}

if [ -n "${TEST_DEFAULT_BUILD}" ]; then
  test_default_build
fi

if [ -n "${TEST_CUSTOM_BUILD_STATIC}" ]; then
  test_custom_build_with_static_dispatch
fi

if [ -n "${TEST_CUSTOM_BUILD_DYNAMIC}" ]; then
  test_custom_build_with_dynamic_dispatch
fi
