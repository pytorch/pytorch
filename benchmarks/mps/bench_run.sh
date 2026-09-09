#!/bin/bash
# Drift-free interleaved A/B softmax bench. For each shape we want A (forced
# MPSGraph) and B (native Metal) measured close in time. The toggle
# (PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX) is read once per process, so each mode
# runs as its own process; we run A then B back-to-back, REPS times, and pair
# them offline (median per cell, via bench_analyze.py) to neutralise drift.
#
# Usage:  PYTHON=python3 REPS=2 OUT=/tmp/bench_out bash benchmarks/mps/bench_run.sh
# Requires this checkout importable as `torch` (pip install -e ., or set PYTHONPATH).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
OUT="${OUT:-/tmp/bench_softmax_out}"
REPS="${REPS:-2}"
FOCUS="${BENCH_FOCUS:-0}"
mkdir -p "$OUT"; rm -f "$OUT"/A_*.json "$OUT"/B_*.json
for r in $(seq 1 "$REPS"); do
  echo "[bench] rep $r mode A (forced MPSGraph)"
  PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX=1 BENCH_MODE=A BENCH_TASK=perf BENCH_FOCUS="$FOCUS" \
    "$PY" "$HERE/bench_softmax.py" > "$OUT/A_$r.json" 2>"$OUT/A_$r.err"
  echo "[bench] rep $r mode B (native Metal)"
  BENCH_MODE=B BENCH_TASK=perf BENCH_FOCUS="$FOCUS" \
    "$PY" "$HERE/bench_softmax.py" > "$OUT/B_$r.json" 2>"$OUT/B_$r.err"
done
echo "[bench] done -> $OUT"
