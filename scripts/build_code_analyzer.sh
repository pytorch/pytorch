#!/bin/bash
##############################################################################
# Build LLVM code analyzer and analyze torch code dependency.
##############################################################################
#
# Example usage:
#
# 1. Analyze torch and generate yaml file of op dependency transitive closure:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# ANALYZE_TORCH=1 scripts/build_code_analyzer.sh
#
# 2. Analyze test project and generate dot file with specific debug filters:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# DEBUG_FILTERS="at::,TypeDefault,CPUType" \
# FORMAT=dot \
# ANALYZE_TEST=1 scripts/build_code_analyzer.sh
#
# 3. Analyze test project and compare with expected result:
# LLVM_DIR=${HOME}/src/llvm8/build/install \
# ANALYZE_TEST=1 CHECK_RESULT=1 scripts/build_code_analyzer.sh

set -ex

SRC_HOME="$( cd "$(dirname "$0")"/.. ; pwd -P)"
ANALYZER_SRC_HOME=${SRC_HOME}/tools/code_analyzer

# Clang/LLVM path
export LLVM_DIR=${LLVM_DIR:-"/usr/lib/llvm-8"}
export CC=${LLVM_DIR}/bin/clang
export CXX=${LLVM_DIR}/bin/clang++

##############################################################################
# Build code analyzer
##############################################################################

BUILD_ROOT=${BUILD_ROOT:-"${SRC_HOME}/build_code_analyzer"}
WORK_DIR=${BUILD_ROOT}/work

mkdir -p ${BUILD_ROOT}
cd ${BUILD_ROOT}
cmake ${ANALYZER_SRC_HOME} -DCMAKE_BUILD_TYPE=Release

if [ -z "${MAX_JOBS}" ]; then
  if [ "$(uname)" == 'Darwin' ]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
  else
    MAX_JOBS=$(nproc)
  fi
fi

make "-j${MAX_JOBS}"

##############################################################################
# Build mobile libtorch into llvm assembly
# ONLY process C++ files for now
##############################################################################

TORCH_BUILD_ROOT=${BUILD_ROOT}/build_mobile
TORCH_INSTALL_PREFIX=${TORCH_BUILD_ROOT}/install

if [ ! -d "${TORCH_INSTALL_PREFIX}" ]; then
  BUILD_ROOT=${TORCH_BUILD_ROOT} CXXFLAGS='-S -emit-llvm' ${SRC_HOME}/scripts/build_mobile.sh
fi

##############################################################################
# Helper function to call LLVM opt
##############################################################################

call_analyzer () {
  OPT_CLOSURE=$([ -n "${CLOSURE}" ] && echo "-closure=${CLOSURE}" || echo "" )
  OPT_DEBUG_FILTERS=$([ -n "${DEBUG_FILTERS}" ] && echo "-df ${DEBUG_FILTERS}" || echo "" )

  echo "Analyze: ${INPUT}"

  ${LLVM_DIR}/bin/opt \
    -load=${BUILD_ROOT}/libOpDependencyPass.so \
    -op_dependency \
    -disable-output \
    -format=${FORMAT} \
    ${OPT_CLOSURE} \
    ${OPT_DEBUG_FILTERS} \
    ${INPUT} > ${OUTPUT}

  echo "Result: ${OUTPUT}"
}

##############################################################################
# Analyze torch code
##############################################################################

if [ -n "${ANALYZE_TORCH}" ]; then
  INPUT=${WORK_DIR}/torch.ll
  FORMAT=${FORMAT:=yaml}
  OUTPUT=${WORK_DIR}/torch_result.${FORMAT}

  if [ ! -f "${INPUT}" ]; then
    # Extract libtorch archive
    OBJECT_DIR=${WORK_DIR}/torch_objs
    rm -rf ${OBJECT_DIR} && mkdir -p ${OBJECT_DIR} && pushd ${OBJECT_DIR}
    ar x ${TORCH_INSTALL_PREFIX}/lib/libc10.a
    ar x ${TORCH_INSTALL_PREFIX}/lib/libtorch.a
    popd

    # Link libtorch into a single module
    ${LLVM_DIR}/bin/llvm-link -S ${OBJECT_DIR}/*.cpp.o -o ${INPUT}
  fi

  # Analyze dependency
  call_analyzer
fi

##############################################################################
# Analyze test project
##############################################################################

if [ -n "${ANALYZE_TEST}" ]; then
  INPUT=${WORK_DIR}/test.ll
  FORMAT=${FORMAT:=yaml}
  OUTPUT=${WORK_DIR}/test_result.${FORMAT}

  # Build test project
  TEST_BUILD_ROOT=${BUILD_ROOT}/build_test
  mkdir -p ${TEST_BUILD_ROOT}
  pushd ${TEST_BUILD_ROOT}
  cmake ${ANALYZER_SRC_HOME}/test -DCMAKE_PREFIX_PATH=${TORCH_INSTALL_PREFIX}
  make "-j${MAX_JOBS}"
  popd

  # Extract archive
  OBJECT_DIR=${WORK_DIR}/test_objs
  rm -rf ${OBJECT_DIR} && mkdir -p ${OBJECT_DIR} && pushd ${OBJECT_DIR}
  ar x ${TORCH_INSTALL_PREFIX}/lib/libc10.a
  ar x ${TEST_BUILD_ROOT}/libSimpleOps.a
  popd

  # Link into a single module
  ${LLVM_DIR}/bin/llvm-link -S ${OBJECT_DIR}/*.cpp.o -o ${INPUT}

  # Analyze dependency
  call_analyzer

  if [ -n "${CHECK_RESULT}" ]; then
    if cmp -s "${OUTPUT}" "${ANALYZER_SRC_HOME}/test/expected_result.yaml"; then
      echo "Test result is the same as expected."
    else
      echo "Test result is DIFFERENT from expected!"
    fi
  fi
fi
