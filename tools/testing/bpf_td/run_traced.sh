#!/usr/bin/env bash
#
# Run a command (typically .ci/pytorch/test.sh) under system-wide bpftrace
# syscall tracing, then post-process the trace into a test-file -> touched-files
# map. Errors never pass silently: if BPF cannot attach we abort before running
# the (expensive) test command.
#
# Usage: run_traced.sh <command> [args...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "run_traced.sh: no command given" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
TRACE_DIR="${PYTORCH_BPF_TD_TRACE_DIR:-${RUNNER_TEMP:-/tmp}/bpf_td}"
mkdir -p "${TRACE_DIR}"
export PYTORCH_BPF_TD_TRACE_DIR="${TRACE_DIR}"

TRACE_OUT="${TRACE_DIR}/trace.out"
TRACE_ERR="${TRACE_DIR}/trace.err"
PIDS_FILE="${TRACE_DIR}/test_pids.jsonl"
: >"${PIDS_FILE}"

maybe_sudo() { if [ "$(id -u)" -ne 0 ]; then sudo "$@"; else "$@"; fi; }

if ! command -v bpftrace >/dev/null 2>&1; then
    echo "run_traced.sh: installing bpftrace via apt" >&2
    maybe_sudo apt-get update -y
    maybe_sudo apt-get install -y bpftrace
fi

echo "run_traced.sh: verifying BPF attach capability" >&2
if ! maybe_sudo timeout 30 bpftrace -e \
    'tracepoint:syscalls:sys_enter_openat { @n = count(); exit(); }' \
    >/dev/null 2>"${TRACE_DIR}/canary.err"; then
    echo "run_traced.sh: FATAL: bpftrace cannot attach probes." >&2
    echo "  The container likely lacks BPF caps (need --privileged or" >&2
    echo "  --cap-add SYS_ADMIN --cap-add BPF --cap-add PERFMON)." >&2
    cat "${TRACE_DIR}/canary.err" >&2 || true
    exit 3
fi

echo "run_traced.sh: starting tracer -> ${TRACE_OUT}" >&2
maybe_sudo bpftrace "${HERE}/trace_opens.bt" >"${TRACE_OUT}" 2>"${TRACE_ERR}" &
TRACER_PID=$!

stop_tracer() {
    if kill -0 "${TRACER_PID}" 2>/dev/null; then
        maybe_sudo kill -TERM "${TRACER_PID}" 2>/dev/null || true
        wait "${TRACER_PID}" 2>/dev/null || true
    fi
}
trap stop_tracer EXIT

# Give bpftrace a moment to attach before the workload starts.
sleep 3

# Trace ALL tests: disable target determination so nothing is skipped.
export NO_TD=1

set +e
"$@"
CMD_RC=$?
set -e

stop_tracer
trap - EXIT

TORCH_ROOT="$(python -c 'import os,torch; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
OUT="${REPO_ROOT}/test/test-reports/bpf_td_mapping_${TEST_CONFIG:-unknown}_${SHARD_NUMBER:-0}.json"

echo "run_traced.sh: building mapping -> ${OUT}" >&2
python "${HERE}/build_mapping.py" \
    --trace "${TRACE_OUT}" \
    --pids "${PIDS_FILE}" \
    --repo-root "${REPO_ROOT}" \
    --torch-root "${TORCH_ROOT}" \
    --out "${OUT}"

exit "${CMD_RC}"
