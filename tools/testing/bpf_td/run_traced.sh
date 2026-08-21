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

# The CI test container is privileged (see bpf-td-trace.yml) but does not
# auto-mount tracefs, and its apt bpftrace is too old (BCC, needs kernel
# headers). Mount tracefs ourselves and use the upstream static bpftrace
# binary (BTF/CO-RE, no headers). This is the recipe proven by the canary.
maybe_sudo mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null || true
maybe_sudo mount -t tracefs tracefs /sys/kernel/tracing 2>/dev/null || true

BPFTRACE_VERSION=v0.26.1
BPFTRACE="$(command -v bpftrace || true)"
if [ -z "${BPFTRACE}" ]; then
    BPFTRACE="${TRACE_DIR}/bpftrace"
    if [ ! -x "${BPFTRACE}" ]; then
        echo "run_traced.sh: fetching static bpftrace ${BPFTRACE_VERSION}" >&2
        # ponytail: ~164MB download per shard; cache in the CI image or a build
        # artifact if this outgrows the experiment.
        curl -fsSL -o "${BPFTRACE}" \
            "https://github.com/bpftrace/bpftrace/releases/download/${BPFTRACE_VERSION}/bpftrace"
        chmod +x "${BPFTRACE}"
    fi
fi

echo "run_traced.sh: verifying BPF attach capability (${BPFTRACE})" >&2
if ! maybe_sudo timeout 30 "${BPFTRACE}" -e \
    'tracepoint:syscalls:sys_enter_openat { @n = count(); exit(); }' \
    >/dev/null 2>"${TRACE_DIR}/canary.err"; then
    echo "run_traced.sh: FATAL: bpftrace cannot attach probes." >&2
    echo "  The container likely lacks BPF caps (need --privileged) or" >&2
    echo "  tracefs is unavailable (need a real docker-on-VM host, not ARC)." >&2
    cat "${TRACE_DIR}/canary.err" >&2 || true
    exit 3
fi

echo "run_traced.sh: starting tracer -> ${TRACE_OUT}" >&2
maybe_sudo "${BPFTRACE}" "${HERE}/trace_opens.bt" >"${TRACE_OUT}" 2>"${TRACE_ERR}" &
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

# Fallback only: build_mapping.py also pattern-matches site-packages/torch, so
# an empty TORCH_ROOT is not fatal. Use python3 (matches the wheel install).
TORCH_ROOT="$(python3 -c 'import os,torch; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
[ -z "${TORCH_ROOT}" ] && echo "run_traced.sh: WARN torch not importable here; relying on path pattern" >&2
OUT="${REPO_ROOT}/test/test-reports/bpf_td_mapping_${TEST_CONFIG:-unknown}_${SHARD_NUMBER:-0}.json"

echo "run_traced.sh: building mapping -> ${OUT}" >&2
python "${HERE}/build_mapping.py" \
    --trace "${TRACE_OUT}" \
    --pids "${PIDS_FILE}" \
    --repo-root "${REPO_ROOT}" \
    --torch-root "${TORCH_ROOT}" \
    --out "${OUT}"

exit "${CMD_RC}"
