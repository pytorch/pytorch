#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[distributed] Source dir: $(pwd)"

source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== Env check (CUDA, 2 cards) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "Injected by action: NNODES=${NNODES:-} NPROC_PER_NODE=${NPROC_PER_NODE:-} RANK=${RANK:-}" \
     "LOCAL_RANK=${LOCAL_RANK:-} WORLD_SIZE=${WORLD_SIZE:-} NODE_RANK=${NODE_RANK:-}" \
     "MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-}"
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT
echo "Cleared rank / rendezvous vars above; tests spawn their own subprocesses"

python --version
(cd /tmp && python - <<'PY'
import sys

import torch

print("torch", torch.__version__, torch.__file__)
print("cuda_available", torch.cuda.is_available())
count = torch.cuda.device_count()
print("device_count", count)
if count < 2:
    sys.exit(
        f"[distributed] Need at least 2 PPUs, found {count}. "
        "All tests here are multi-card and would just be skipped (false pass), so failing. "
        "Check that the workflow nproc_per_node is 2 and the pod really got 2 cards."
    )
PY
)
ppu-smi || echo "[warn] ppu-smi unavailable; check that the pod has PPU devices"
bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

K_FP8_EXPR="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2 \
and not scaled_matmul \
and not test_fixed_striding"

K_SKIP_CASES="not test_dtensor_seq_par_shard_dim_0 \
and not test_set_reduce_scatter_divide_factor \
and not test_basic_all_gather_bucketing \
and not test_schedule_overlap_benchmark \
and not test_bucket_exposed_with_hidden_single_overlap \
and not test_bucketing_split_for_overlap \
and not test_bucketing_wait_sink \
and not test_fully_shard_force_sum_reduce_scatter"

K_FP8="$(ppu_cuda_only_k_expr "$K_FP8_EXPR" "$K_SKIP_CASES")"
FAILED_CASES=()
run_case() {
    local own_k="$1"
    shift
    local k_expr label
    if [ -n "$own_k" ]; then
        k_expr="( ${own_k} ) and ( ${K_FP8} )"
    else
        k_expr="${K_FP8}"
    fi
    label="$*${own_k:+ -k ${own_k}}"

    echo "=== [case] ${label} ==="
    if python test/run_test.py --include "$@" -k "$k_expr" --verbose; then
        echo "=== [pass] ${label} ==="
    else
        echo "::error::[fail] ${label}"
        FAILED_CASES+=("${label}")
    fi
}

echo "=== CUDA distributed tests (2 cards; CUDA only, FP8 excluded) ==="

run_case "" distributed/test_c10d_functional_native
run_case "" distributed/tensor/test_dtensor_compile
run_case "" distributed/tensor/parallel/test_micro_pipeline_tp
run_case "" distributed/_composable/test_replicate_with_compiler

run_case "test_train_parity_multi_group"                   distributed/_composable/fsdp/test_fully_shard_training
run_case "test_train_parity_with_activation_checkpointing" distributed/_composable/fsdp/test_fully_shard_training
run_case "test_train_parity_hsdp"                          distributed/_composable/fsdp/test_fully_shard_training
run_case "test_gradient_accumulation"                      distributed/_composable/fsdp/test_fully_shard_training
run_case "test_dp_state_dict_save_load" distributed/_composable/fsdp/test_fully_shard_state_dict
run_case "test_compute_dtype" distributed/_composable/fsdp/test_fully_shard_mixed_precision
run_case "test_reduce_dtype"  distributed/_composable/fsdp/test_fully_shard_mixed_precision
run_case "test_clip_grad_norm" distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_

run_case "" \
    distributed/test_inductor_collectives \
    distributed/test_aten_comm_compute_reordering \
    distributed/test_compute_comm_reordering

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== Failed (${#FAILED_CASES[@]} cases) ==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[distributed] ${#FAILED_CASES[@]} cases failed; see the [case] sections above"
    exit 1
fi

echo "[distributed] Done, all passed"
