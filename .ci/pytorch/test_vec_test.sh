#!/usr/bin/env bash
# Exercise CI dispatch without requiring a PyTorch build or SIMD hardware.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
sed -n '/^test_vec() {$/,/^}$/p' "$repo_root/.ci/pytorch/test.sh" > "$scratch/function.sh"
source "$scratch/function.sh"
assert_git_not_dirty() { :; }

mkdir -p "$scratch/bin" "$scratch/test" "$scratch/build/bin" "$scratch/wheel"
cat > "$scratch/bin/python" <<'PYTHON'
#!/usr/bin/env bash
set -eu
if [[ "$1" == -c ]]; then
  [[ -z "${ATEN_CPU_CAPABILITY+x}" ]] || exit 90
  [[ "${FAIL_PROBE:-0}" == 0 ]] || exit 91
  echo "$FAKE_CAPABILITY"
else
  [[ "$*" == 'test/run_test.py --cpp --verbose -i cpp/vec_test_all_types_'* ]] || exit 92
  [[ "$NO_TD" == 1 ]] || exit 93
  [[ "$CPP_TESTS_DIR" == "$EXPECTED_BIN_DIR" ]] || exit 94
  name="${@: -1}"
  echo "$name" >> "$RUN_LOG"
  [[ "$name" != "cpp/vec_test_all_types_${FAIL_CAPABILITY:-none}" ]]
fi
PYTHON
chmod +x "$scratch/bin/python"
export PATH="$scratch/bin:$PATH"
export BUILD_BIN_DIR="$scratch/build/bin" TORCH_TEST_DIR="$scratch/wheel"
export TORCH_LIB_DIR="$scratch/lib" LD_LIBRARY_PATH=""
export RUN_LOG="$scratch/runs" EXPECTED_BIN_DIR="$BUILD_BIN_DIR"
export FAKE_CAPABILITY=DEFAULT ATEN_CPU_CAPABILITY=default
export BUILD_ENVIRONMENT=linux-cpu TEST_CONFIG=default IN_WHEEL_TEST=""
cd "$scratch"

make_binary() {
  printf '#!/usr/bin/env bash\nexit 0\n' > "$1/vec_test_all_types_$2"
  chmod +x "$1/vec_test_all_types_$2"
}

run_case() {
  local expected=$1 status=0
  : > "$RUN_LOG"
  test_vec > "$scratch/output" 2>&1 || status=$?
  if [[ "$status" != "$expected" ]]; then
    cat "$scratch/output"
    echo "Expected exit $expected, got $status" >&2
    exit 1
  fi
}

# DEFAULT must exist; a missing test directory must not pass either.
run_case 1
grep -q 'Missing required vec test' "$scratch/output"
export BUILD_BIN_DIR="$scratch/missing"
run_case 1
grep -q 'Missing required vec test' "$scratch/output"
export BUILD_BIN_DIR="$scratch/build/bin"
make_binary "$BUILD_BIN_DIR" DEFAULT
run_case 0
[[ $(wc -l < "$RUN_LOG") -eq 1 ]]

# Probe the hardware despite ATEN_CPU_CAPABILITY=default, with no duplicates.
export FAKE_CAPABILITY=AVX512
make_binary "$BUILD_BIN_DIR" AVX2
make_binary "$BUILD_BIN_DIR" AVX512
run_case 0
[[ $(wc -l < "$RUN_LOG") -eq 3 ]]
for capability in DEFAULT AVX2 AVX512; do
  grep -qx "cpp/vec_test_all_types_$capability" "$RUN_LOG"
done

# A failure must propagate, but must not stop the other capabilities running.
export FAIL_CAPABILITY=AVX2
run_case 1
[[ $(wc -l < "$RUN_LOG") -eq 3 ]]
unset FAIL_CAPABILITY
rm "$BUILD_BIN_DIR/vec_test_all_types_AVX512"
run_case 0
grep -q 'Warning:.*AVX512.*missing' "$scratch/output"
[[ $(wc -l < "$RUN_LOG") -eq 2 ]]

export FAIL_PROBE=1
run_case 1
[[ ! -s "$RUN_LOG" ]]
unset FAIL_PROBE

# Wheel tests use the installed test binaries, not the build tree.
export IN_WHEEL_TEST=1 EXPECTED_BIN_DIR="$TORCH_TEST_DIR" FAKE_CAPABILITY=DEFAULT
run_case 1
make_binary "$TORCH_TEST_DIR" DEFAULT
run_case 0
[[ $(wc -l < "$RUN_LOG") -eq 1 ]]
export IN_WHEEL_TEST="" EXPECTED_BIN_DIR="$BUILD_BIN_DIR"

export FAKE_CAPABILITY='Z VECTOR'
make_binary "$BUILD_BIN_DIR" ZVECTOR
run_case 0
grep -qx 'cpp/vec_test_all_types_ZVECTOR' "$RUN_LOG"
export FAKE_CAPABILITY=SVE256
make_binary "$BUILD_BIN_DIR" SVE256
run_case 0
grep -qx 'cpp/vec_test_all_types_SVE256' "$RUN_LOG"

for BUILD_ENVIRONMENT in linux-cuda linux-rocm linux-xpu; do
  export BUILD_ENVIRONMENT
  run_case 0
  [[ ! -s "$RUN_LOG" ]]
done
export BUILD_ENVIRONMENT=linux-cpu TEST_CONFIG=slow
run_case 0
[[ ! -s "$RUN_LOG" ]]
echo 'test_vec dispatch regression checks passed'
