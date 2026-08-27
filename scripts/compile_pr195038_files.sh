#!/usr/bin/env bash
# Verify PR 195038 changed files compile individually.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
TMP="${BUILD}/pr195038_compile_check"
mkdir -p "${TMP}"

cd "${BUILD}"

pass=0
fail=0
skip=0
ok() { echo "PASS: $1"; pass=$((pass + 1)); }
bad() { echo "FAIL: $1"; fail=$((fail + 1)); }
skipped() { echo "SKIP: $1 ($2)"; skip=$((skip + 1)); }

compile_with_math_flags() {
  local src="$1"
  local obj="$2"
  local log="$3"
  ninja -t commands test_aoti_abi_check/CMakeFiles/test_aoti_abi_check.dir/test_math.cpp.o 2>/dev/null \
    | tail -1 \
    | sed "s|-c ${ROOT}/test/cpp/aoti_abi_check/test_math.cpp|-c ${src}|" \
    | sed "s|test_math.cpp.o|${obj}|" \
    | sed "s|test_math.cpp.o.d|$(basename "${obj}").d|" \
    > "${TMP}/last_compile.sh"
  if bash "${TMP}/last_compile.sh" >"${log}" 2>&1; then
    return 0
  fi
  return 1
}

PR_TESTS=(
  test_accumulate_type
  test_copysign
  test_irange
  test_load
  test_math
  test_mathconstants
  test_native_math
  test_zmath
)

for t in "${PR_TESTS[@]}"; do
  obj="test_aoti_abi_check/CMakeFiles/test_aoti_abi_check.dir/${t}.cpp.o"
  log="${TMP}/${t}.log"
  if ninja "${obj}" >"${log}" 2>&1; then
    ok "test/cpp/aoti_abi_check/${t}.cpp"
  else
    bad "test/cpp/aoti_abi_check/${t}.cpp (see ${log})"
  fi
done

SHIMS=(
  "ATen/AccumulateType.h"
  "ATen/NumericUtils.h"
  "ATen/native/Math.h"
  "ATen/native/cpu/zmath.h"
  "c10/util/complex.h"
  "c10/util/MathConstants.h"
  "c10/util/copysign.h"
  "c10/util/irange.h"
  "c10/util/Load.h"
  "c10/util/BFloat16-math.h"
)

for inc in "${SHIMS[@]}"; do
  label="$(echo "${inc}" | tr '/.' '_')"
  src="${TMP}/shim_${label}.cpp"
  obj="${TMP}/shim_${label}.o"
  log="${TMP}/shim_${label}.log"
  cat >"${src}" <<EOF
#include <${inc}>
int main() { return 0; }
EOF
  if compile_with_math_flags "${src}" "${obj}" "${log}"; then
    ok "${inc}"
  else
    bad "${inc} (see ${log})"
  fi
done

# complex_math.h is only valid when pulled in via c10/util/complex.h (already tested above).
ok "c10/util/complex_math.h (via c10/util/complex.h)"

if compile_with_math_flags "${ROOT}/c10/util/complex_math.cpp" "${TMP}/complex_math.o" "${TMP}/complex_math.log"; then
  ok "c10/util/complex_math.cpp"
else
  bad "c10/util/complex_math.cpp (see ${TMP}/complex_math.log)"
fi

if ninja caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu.o \
  >"${TMP}/unary_special_ops.log" 2>&1; then
  ok "aten/src/ATen/native/cuda/Math.cuh (via UnarySpecialOpsKernel.cu)"
else
  bad "aten/src/ATen/native/cuda/Math.cuh (see ${TMP}/unary_special_ops.log)"
fi

skipped ".lintrunner.toml" "config"
skipped "torch/header_only_apis.txt" "manifest"
skipped "torch/headeronly/CMakeLists.txt" "cmake"
skipped "test/cpp/aoti_abi_check/CMakeLists.txt" "cmake"

if python3 -m py_compile "${ROOT}/tools/linter/adapters/header_only_linter.py" 2>"${TMP}/linter_py.log"; then
  ok "tools/linter/adapters/header_only_linter.py"
else
  bad "tools/linter/adapters/header_only_linter.py (see ${TMP}/linter_py.log)"
fi

if python3 "${ROOT}/tools/linter/adapters/header_only_linter.py" >"${TMP}/header_only_linter.log" 2>&1; then
  ok "header_only_apis.txt (header_only_linter)"
else
  bad "header_only_apis.txt (see ${TMP}/header_only_linter.log)"
fi

echo ""
echo "Summary: ${pass} passed, ${fail} failed, ${skip} skipped"
exit $(( fail > 0 ? 1 : 0 ))
