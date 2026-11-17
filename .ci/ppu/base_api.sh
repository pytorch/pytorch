#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[base_api] Source dir: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_test_deps.sh

CRASH_LIST="${CRASH_LIST:-.ci/ppu/badcase_list/base_api.txt}"
echo "[base_api] crash list: ${CRASH_LIST}"

FAILED_CASES=()
run_pytest() {
    local test_file="$1"
    local k_expr="${2:-}"
    echo "=== [case] ${test_file}${k_expr:+  -k ${k_expr}} ==="
    if [[ -n "$k_expr" ]]; then
        if python .ci/ppu/multi_run_case.py "$test_file" --crash-list "$CRASH_LIST" -k "$k_expr"; then
            echo "=== [pass] ${test_file} ==="
        else
            echo "::error::[fail] ${test_file}"
            FAILED_CASES+=("${test_file}")
        fi
    else
        if python .ci/ppu/multi_run_case.py "$test_file" --crash-list "$CRASH_LIST"; then
            echo "=== [pass] ${test_file} ==="
        else
            echo "::error::[fail] ${test_file}"
            FAILED_CASES+=("${test_file}")
        fi
    fi
}

echo "=== CUDA base api tests (test_ops / test_ops_gradients, CUDA only) ==="

run_pytest test/test_ops.py "cuda"
run_pytest test/test_ops_gradients.py "cuda"

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== Failed (${#FAILED_CASES[@]} files) ==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[base_api] ${#FAILED_CASES[@]} test files failed; see the [case] sections above"
    exit 1
fi

echo "[base_api] Done, all passed"
