#!/usr/bin/env bash
# Run ncclEp unit tests across multiple GPUs.
#
# Spawns one bash process per rank in the background. Each test binary picks
# the GPU via cudaSetDevice(rank % device_count). NCCL bootstrap uses a
# shared UID file: rank 0 writes, all other ranks poll for it (implemented
# in test_common.h::exchange_uid). No MPI runtime required.
#
# Usage:
#   NCCL_HOME=/path/to/nccl/build NCCL_EP_BUILDDIR=/path/to/nccl_ep/build bash run_tests.sh [num_gpus]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_HOME="${NCCL_HOME:-$(cd "${SCRIPT_DIR}/../../../build" && pwd)}"
NCCL_EP_BUILDDIR="${NCCL_EP_BUILDDIR:-${NCCL_HOME}}"
NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

export LD_LIBRARY_PATH="${NCCL_EP_BUILDDIR}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"

GTEST_ARGS="${GTEST_FILTER:+--gtest_filter=${GTEST_FILTER}}"
OVERALL_FAIL=0

run_suite() {
    local BINARY="$1"
    local SUITE_NAME="$2"
    local MIN_GPUS="${3:-4}"
    local TEST_BIN="${NCCL_EP_BUILDDIR}/test/nccl_ep/${BINARY}"

    if [[ ! -x "${TEST_BIN}" ]]; then
        echo "ERROR: binary not found: ${TEST_BIN}"
        echo "Build first:  make -C ${SCRIPT_DIR} NCCL_HOME=${NCCL_HOME} NCCL_EP_BUILDDIR=${NCCL_EP_BUILDDIR}"
        return 1
    fi

    if (( NUM_GPUS < MIN_GPUS )); then
        echo "${SUITE_NAME}: requires at least ${MIN_GPUS} GPUs, found ${NUM_GPUS}. Skipping."
        return 0
    fi

    local TMPDIR_L="${TMPDIR:-/tmp}"
    local UID_FILE="${TMPDIR_L}/te_ep_uid_${BINARY}_$$"
    rm -f "${UID_FILE}"
    trap "rm -f '${UID_FILE}'" EXIT INT TERM

    local LOG_DIR
    LOG_DIR=$(mktemp -d)
    local FAIL=0

    echo "=== ${SUITE_NAME} ==="
    echo "  GPUs: ${NUM_GPUS}   Binary: ${TEST_BIN}"
    echo

    # Spawn one process per rank. GPU binding is handled inside the test
    # binary via cudaSetDevice(rank % device_count).
    local PIDS=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        "${TEST_BIN}" \
            --rank="${i}" \
            --nranks="${NUM_GPUS}" \
            --uid-file="${UID_FILE}" \
            ${GTEST_ARGS} \
            > "${LOG_DIR}/rank_${i}.log" 2>&1 &
        PIDS+=($!)
    done
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        wait "${PIDS[$i]}" || FAIL=1
    done

    echo "--- Rank 0 output ---"
    cat "${LOG_DIR}/rank_0.log"

    if (( FAIL )); then
        for i in $(seq 1 $((NUM_GPUS - 1))); do
            echo "--- Rank ${i} output ---"
            cat "${LOG_DIR}/rank_${i}.log"
        done
        echo "=== ${SUITE_NAME}: FAILED ==="
        OVERALL_FAIL=1
    else
        echo "=== ${SUITE_NAME}: ALL PASSED ==="
    fi

    rm -rf "${LOG_DIR}"
}

run_suite "test_output_layout" "EP Output Layout Tests"
run_suite "test_handle_maps"   "EP Handle Maps Tests"
run_suite "test_lifecycle"     "EP Lifecycle Tests"
run_suite "test_ht_bwd"        "EP HT Backward Tests"
run_suite "test_tensor_create" "EP Tensor Create Tests"
run_suite "test_ht_stale_routing_map" "EP HT Stale Routing Map Tests"

exit "${OVERALL_FAIL}"
