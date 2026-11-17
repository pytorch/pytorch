#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PPU_BOARD="${PPU_BOARD:-}"
PPU_BRANCH="${PPU_BRANCH:-unknown}"
NAS_LOG_DIR="${NAS_LOG_DIR:-/mnt/wl_nas/ut_test/logs}"
SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"
TEE_WAIT_SECONDS="${TEE_WAIT_SECONDS:-60}"

copy_log_to_nas() {
    local rc=$?
    trap '' INT TERM
    exec 1>&3 2>&4
    local waited=0
    while kill -0 "$TEE_PID" 2>/dev/null && (( waited < TEE_WAIT_SECONDS )); do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$TEE_PID" 2>/dev/null; then
        echo "::warning::[full_ut] tee did not exit within ${TEE_WAIT_SECONDS}s (leftover processes may hold the pipe), killing it" >&2
        kill "$TEE_PID" 2>/dev/null || true
    fi
    wait "$TEE_PID" 2>/dev/null || true
    local nas_file="$NAS_LOG_FILE" suffix=2
    while [[ -e "$nas_file" ]]; do
        nas_file="${NAS_LOG_FILE%.txt}_${suffix}.txt"
        suffix=$((suffix + 1))
    done
    if mkdir -p "$NAS_LOG_DIR" && cp -- "$LOCAL_LOG_FILE" "$nas_file"; then
        echo "[full_ut] Log copied to NAS: ${nas_file}"
    else
        echo "::warning::[full_ut] Failed to copy log to NAS: ${LOCAL_LOG_FILE} -> ${nas_file}" >&2
    fi
    exit "$rc"
}

if [[ -n "$PPU_BOARD" ]]; then
    LOG_NAME="$(TZ=UTC-8 date +%Y_%m_%d_%H)_${PPU_BRANCH//\//-}_full_ut_${PPU_BOARD}_shard${SHARD_NUMBER}.txt"
    NAS_LOG_FILE="${NAS_LOG_DIR}/${LOG_NAME}"
    LOCAL_LOG_FILE="${TMPDIR:-/tmp}/${LOG_NAME}"
    : >"$LOCAL_LOG_FILE"
    exec 3>&1 4>&2
    exec > >(tee -a "$LOCAL_LOG_FILE") 2>&1
    TEE_PID=$!
    trap copy_log_to_nas EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo "[full_ut] Log will be copied to NAS on exit: ${NAS_LOG_FILE}"
else
    echo "::warning::[full_ut] PPU_BOARD not set, skipping log archiving" >&2
fi
echo "[full_ut] Source dir: $(pwd)"

NAS_LIST_DIR="${NAS_LIST_DIR:-/mnt/wl_nas/ut_test/pytorch}"
NAS_UT_FILES="${NAS_UT_FILES:-$NAS_LIST_DIR/ut_files.txt}"
NAS_SKIP_FILE="${NAS_SKIP_FILE:-$NAS_LIST_DIR/full_ut_skip.txt}"
LOCAL_LIST_DIR=".ci/ppu/badcase_list"
LOCAL_UT_FILES="$LOCAL_LIST_DIR/ut_files.txt"
LOCAL_SKIP_FILE="$LOCAL_LIST_DIR/full_ut_skip.txt"
UT_FILES="${UT_FILES:-$LOCAL_UT_FILES}"
CRASH_LIST="${CRASH_LIST:-$LOCAL_SKIP_FILE}"

copy_nas_list() {
    local label="$1" src="$2" dst="$3"
    if [[ ! -f "$src" || ! -r "$src" ]]; then
        echo "::error::[full_ut] NAS ${label} missing or unreadable: ${src}" >&2
        exit 1
    fi
    if ! cp -- "$src" "$dst"; then
        echo "::error::[full_ut] Failed to copy NAS ${label}: ${src} -> ${dst}" >&2
        exit 1
    fi
    echo "[full_ut] Copied NAS ${label}: ${src} -> ${dst}"
}

mkdir -p "$LOCAL_LIST_DIR"
copy_nas_list "test file list" "$NAS_UT_FILES" "$LOCAL_UT_FILES"
copy_nas_list "skip list" "$NAS_SKIP_FILE" "$LOCAL_SKIP_FILE"

TEST_FILES_RAW="$(python - "$UT_FILES" "$CRASH_LIST" "$SHARD_NUMBER" "$NUM_TEST_SHARDS" <<'PY'
import re
import sys

ut_files, skip_file, shard_number, num_shards = sys.argv[1:]
if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in (shard_number, num_shards)):
    sys.exit("[full_ut] SHARD_NUMBER and NUM_TEST_SHARDS must be positive integers")
shard_number, num_shards = int(shard_number), int(num_shards)
if shard_number > num_shards:
    sys.exit("[full_ut] SHARD_NUMBER must not exceed NUM_TEST_SHARDS")

with open(ut_files) as f:
    test_files = [text for line in f if (text := line.split("#", 1)[0].strip())]
if not test_files:
    sys.exit("[full_ut] Test file list is empty")

with open(skip_file) as f:
    skipped_files = {
        text for line in f
        if (text := line.partition("\t")[0].strip())
        and not text.startswith("#") and "::" not in text and text.endswith(".py")
    }
selected_files = []
for test_file in test_files:
    if test_file in skipped_files:
        print(f"=== [skip] {test_file} (whole file) ===", file=sys.stderr)
    else:
        selected_files.append(test_file)

for index, test_file in enumerate(selected_files):
    if index % num_shards == shard_number - 1:
        print(test_file)
PY
)"

source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh
bash .ci/ppu/install_test_deps.sh

FAILED_CASES=()
run_pytest() {
    local test_file="$1"
    echo "=== [case] ${test_file}  -k cuda ==="
    if python .ci/ppu/multi_run_case.py "$test_file" --crash-list "$CRASH_LIST" -k "cuda"; then
        echo "=== [done] ${test_file} (passed or skipped) ==="
    else
        echo "::error::[fail] ${test_file}"
        FAILED_CASES+=("${test_file}")
    fi
}

echo "=== Full UT (shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS}) ==="
TEST_FILE_COUNT=0
if [[ -n "$TEST_FILES_RAW" ]]; then
    while IFS= read -r test_file; do
        TEST_FILE_COUNT=$((TEST_FILE_COUNT + 1))
        run_pytest "$test_file"
    done <<<"$TEST_FILES_RAW"
fi

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== Failed (${#FAILED_CASES[@]} files) ==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[full_ut] ${#FAILED_CASES[@]} test files failed; see the [case] sections above"
    exit 1
fi

echo "[full_ut] Done, shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS}: all ${TEST_FILE_COUNT} test files passed or skipped"