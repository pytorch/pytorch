#!/usr/bin/env bash
# NCCL EP CI: run ep_test and ep_bench inside a Slurm allocation (srun --mpi=pmix, SRUN_MPI-style).
# Set OMPI_MCA_* in CI for EOS TCP BTL (e.g. btl_tcp_if_include) when applicable.
# NCCL_EP_BENCH_HT=1 enables ep_bench --algorithm high-throughput (optional; not all clusters pass it).

: "${NCCL_HOME:?NCCL_HOME must be set}"
NCCL_EP_BUILDDIR="${NCCL_EP_BUILDDIR:-${NCCL_HOME}}"

EP_TEST="${NCCL_EP_BUILDDIR}/test/nccl_ep/ep_test"
EP_BENCH="${NCCL_EP_BUILDDIR}/test/nccl_ep/ep_bench"
if [[ ! -x "$EP_TEST" ]]; then
  echo "ERROR: ep_test not found or not executable: $EP_TEST" >&2
  exit 1
fi
if [[ ! -x "$EP_BENCH" ]]; then
  echo "ERROR: ep_bench not found or not executable: $EP_BENCH" >&2
  exit 1
fi

: "${NP:?NP must be set}"
# Set SLURM_PARTITION in CI for IPP6; leave unset on EOS (default partition / allocation).
PARTITION="${SLURM_PARTITION:-}"
TIME="${NCCL_EP_TEST_SLURM_TIME:-00:30:00}"
BENCH_TIME="${NCCL_EP_BENCH_SLURM_TIME:-$TIME}"

export LD_LIBRARY_PATH="${NCCL_EP_BUILDDIR}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

# GDAKI (GPU Direct Async Kernel-Initiated) for multi-node RDMA / GIN
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"

set -e

run_nccl_ep_srun() {
  local binary="$1"
  local walltime="$2"
  shift 2
  set -x
  # shellcheck disable=SC2086
  srun \
    ${PARTITION:+-p "${PARTITION}"} \
    ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
    -N 1 \
    -n "${NP}" \
    --exclusive \
    -t "${walltime}" \
    --mpi=pmix \
    --cpu-bind=none \
    --export=ALL \
    ${NCCL_EP_SRUN_EXTRA:-} \
    "$binary" "$@"
  set +x
}

# Run the two extra token-distribution variants for ep_bench: --tokens stays at
# $tokens (= buffer cap) across all calls so kernel iteration counts are identical
# and per-variant timings are directly comparable.
run_ep_bench_variants() {
  local algorithm="$1"
  local tokens="$2"
  local less_than=$((tokens / 2))
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm "$algorithm" --tokens "$tokens" --hidden 7168 --top-k 8 --experts 256 --validate --dispatch-less-than-max-tokens "$less_than"
  run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" --algorithm "$algorithm" --tokens "$tokens" --hidden 7168 --top-k 8 --experts 256 --validate --non-uniform-tokens
}

# ep_test: low-latency (expert-major, LL-hardcoded) / high-throughput × {flat, expert-major}
# ep_test is kept at one canonical batch size per algorithm (smoke coverage); batch-size
# sweep lives in ep_bench below.
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ll              -t 128  -d 7168
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ht -L fl        -t 4096 -d 7168
run_nccl_ep_srun "$EP_TEST" "$TIME" -a ht -L em        -t 4096 -d 7168

# ep_bench: layout × batch-size cross-product (override bench wall time with NCCL_EP_BENCH_SLURM_TIME if needed)
# LL supports {expert-major, rank-major}; HT supports {flat, expert-major}.
# Batch sizes (tokens per rank): 128, 256, 1K, 4K, 8K.
EP_BENCH_TOKEN_SIZES=(128 256 1024 4096)

run_ep_bench_layout_size_sweep() {
  local algorithm="$1"; shift
  local layouts=("$@")
  local layout tokens
  for layout in "${layouts[@]}"; do
    for tokens in "${EP_BENCH_TOKEN_SIZES[@]}"; do
      run_nccl_ep_srun "$EP_BENCH" "$BENCH_TIME" \
        --algorithm "$algorithm" --layout "$layout" \
        --tokens "$tokens" --hidden 7168 --top-k 8 --experts 256 --validate
    done
  done
}

run_ep_bench_layout_size_sweep low-latency em rm
# Token-distribution variants stay at the canonical LL batch size (cover the variant axis,
# not the size axis — already swept above).
run_ep_bench_variants low-latency 128

# High-throughput ep_bench (set NCCL_EP_BENCH_HT=1 to run; off by default — cluster-dependent)
if [[ "${NCCL_EP_BENCH_HT:-0}" == "1" ]]; then
  run_ep_bench_layout_size_sweep high-throughput fl em
  run_ep_bench_variants high-throughput 4096
fi
