#!/bin/bash
##############################################################################
# Build LLVM code analyzer and analyze torch code dependency.
##############################################################################
#
# Example usage:
#
# 1. Analyze torch and generate yaml file of op dependency transitive closure:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# ANALYZE_TORCH=1 tools/code_analyzer/build.sh
#
# 2. Analyze test project and compare with expected result:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# ANALYZE_TEST=1 tools/code_analyzer/build.sh
#
# 3. Analyze torch and generate yaml file of op dependency with debug path:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# ANALYZE_TORCH=1 tools/code_analyzer/build.sh -debug_path=true
#
# If you're a Facebook employee, chances are you're running on CentOS 8.
# If that's the case, you can install all the dependencies you need with:
#
#   sudo dnf install llvm-devel llvm-static clang ncurses-devel
#
# and then set LLVM_DIR=/usr

set -ex

SRC_ROOT="$( cd "$(dirname "$0")"/../.. ; pwd -P)"
ANALYZER_SRC_HOME="${SRC_ROOT}/tools/code_analyzer"

# Clang/LLVM path
export LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-8}"
export CC="${LLVM_DIR}/bin/clang"
export CXX="${LLVM_DIR}/bin/clang++"
EXTRA_ANALYZER_FLAGS=$@

BUILD_ROOT="${BUILD_ROOT:-${SRC_ROOT}/build_code_analyzer}"
WORK_DIR="${BUILD_ROOT}/work"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"
mkdir -p "${WORK_DIR}"
cd "${BUILD_ROOT}"

build_analyzer() {
  cmake "${ANALYZER_SRC_HOME}" -DCMAKE_BUILD_TYPE=Release

  if [ -z "${MAX_JOBS}" ]; then
    if [ "$(uname)" == 'Darwin' ]; then
      MAX_JOBS=$(sysctl -n hw.ncpu)
    else
      MAX_JOBS=$(nproc)
    fi
  fi

  make "-j${MAX_JOBS}"
}

build_torch_mobile() {
  TORCH_BUILD_ROOT="${BUILD_ROOT}/build_mobile"
  TORCH_INSTALL_PREFIX="${TORCH_BUILD_ROOT}/install"

  BUILD_ROOT="${TORCH_BUILD_ROOT}" "${SRC_ROOT}/scripts/build_mobile.sh" \
    -DCMAKE_CXX_FLAGS="-S -emit-llvm -DSTRIP_ERROR_MESSAGES" \
    ${MOBILE_BUILD_FLAGS}
}

build_test_project() {
  TEST_SRC_ROOT="${SRC_ROOT}/test/mobile/op_deps"
  TEST_BUILD_ROOT="${BUILD_ROOT}/build_test"
  TEST_INSTALL_PREFIX="${TEST_BUILD_ROOT}/install"

  BUILD_ROOT="${TEST_BUILD_ROOT}" \
    TORCH_INSTALL_PREFIX="${TORCH_INSTALL_PREFIX}" \
    "${TEST_SRC_ROOT}/build.sh" \
    -DCMAKE_CXX_FLAGS="-S -emit-llvm -DSTRIP_ERROR_MESSAGES"
}

call_analyzer() {
  ANALYZER_BIN="${BUILD_ROOT}/analyzer" \
    INPUT="${INPUT}" OUTPUT="${OUTPUT}" \
    EXTRA_ANALYZER_FLAGS="${EXTRA_ANALYZER_FLAGS}" \
    "${ANALYZER_SRC_HOME}/run_analyzer.sh"
}

analyze_torch_mobile() {
  INPUT="${WORK_DIR}/torch.ll"
  OUTPUT="${WORK_DIR}/torch_result.yaml"

  if [ ! -f "${INPUT}" ]; then
    # Link libtorch into a single module
    # TODO: invoke llvm-link from cmake directly to avoid this hack.
    # TODO: include *.c.o when there is meaningful fan-out from pure-c code.
    "${LLVM_DIR}/bin/llvm-link" -S \
    $(find "${TORCH_BUILD_ROOT}" -name '*.cpp.o' -o -name '*.cc.o') \
    -o "${INPUT}"
  fi

  # Analyze dependency
  call_analyzer
}

print_output_file_path() {
  echo "Deployed file at: ${OUTPUT}"
}

analyze_test_project() {
  INPUT="${WORK_DIR}/test.ll"
  OUTPUT="${WORK_DIR}/test_result.yaml"

  # Link into a single module (only need c10 and OpLib srcs)
  # TODO: invoke llvm-link from cmake directly to avoid this hack.
  "${LLVM_DIR}/bin/llvm-link" -S \
  $(find "${TORCH_BUILD_ROOT}" -path '*/c10*' \( -name '*.cpp.o' -o -name '*.cc.o' \)) \
  $(find "${TEST_BUILD_ROOT}" -path '*/OpLib*' \( -name '*.cpp.o' -o -name '*.cc.o' \)) \
  -o "${INPUT}"

  # Analyze dependency
  call_analyzer
}

check_test_result() {
  if cmp -s "${OUTPUT}" "${TEST_SRC_ROOT}/expected_deps.yaml"; then
    echo "Test result is the same as expected."
  else
    echo "Test result is DIFFERENT from expected!"
    diff -u "${TEST_SRC_ROOT}/expected_deps.yaml" "${OUTPUT}"
    exit 1
  fi
}

build_analyzer

if [ -n "${ANALYZE_TORCH}" ]; then
  build_torch_mobile
  analyze_torch_mobile
  if [ -n "${DEPLOY}" ]; then
    print_output_file_path
  fi
fi

if [ -n "${ANALYZE_TEST}" ]; then
  build_torch_mobile
  build_test_project
  analyze_test_project
  check_test_result
fi
