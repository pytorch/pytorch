import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class GluonGroupedMMConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    NUM_LOAD_BUFFERS: int
    NUM_ACC_BUFFERS: int
    NUM_STORE_WARPS: int = 4


def compute_stage_variants_gluon(
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype,
    tmem_max_columns: int = 512,
    max_configs: int = 1,
):
    """
    Compute valid (NUM_LOAD_BUFFERS, NUM_ACC_BUFFERS) pairs for given block
    dimensions, sampled evenly across the whole valid range so the result
    isn't biased toward the largest NUM_LOAD_BUFFERS that happens to fit.
    Returns at most max_configs pairs.
    """
    import torch

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    smem_limit = 227 * 1024  # hardware limit

    a_bytes_per_stage = BLOCK_M * BLOCK_K * dtype_bytes
    b_bytes_per_stage = BLOCK_N * BLOCK_K * dtype_bytes
    c_bytes_per_stage = BLOCK_M * BLOCK_N * dtype_bytes
    ab_bytes_per_stage = a_bytes_per_stage + b_bytes_per_stage

    min_load_buffers = 1
    min_acc_buffers = 1
    compiler_overhead = 256

    min_smem = (
        ab_bytes_per_stage * min_load_buffers
        + c_bytes_per_stage
        + 8 * min_load_buffers * 2
        + 8 * min_acc_buffers * 2
        + compiler_overhead
    )

    if min_smem > smem_limit:
        return []

    all_valid = []
    for num_load_buffers in range(8, 0, -1):
        ab_smem = ab_bytes_per_stage * num_load_buffers
        c_smem = c_bytes_per_stage
        load_barrier_smem = 8 * num_load_buffers * 2

        base_smem = ab_smem + c_smem + load_barrier_smem + compiler_overhead

        if base_smem > smem_limit:
            continue

        max_acc_by_tmem = tmem_max_columns // BLOCK_N
        remaining_smem = smem_limit - base_smem
        max_acc_by_smem = remaining_smem // (8 * 2)

        max_acc_buffers = min(max_acc_by_tmem, max_acc_by_smem, 8)

        for num_acc_buffers in range(max_acc_buffers, 0, -1):
            acc_barrier_smem = 8 * num_acc_buffers * 2
            total_smem = base_smem + acc_barrier_smem
            tmem_cols = BLOCK_N * num_acc_buffers

            if total_smem <= smem_limit and tmem_cols <= tmem_max_columns:
                all_valid.append((num_load_buffers, num_acc_buffers))

    if len(all_valid) <= max_configs:
        return all_valid

    stride = len(all_valid) / max_configs
    return [all_valid[int(i * stride)] for i in range(max_configs)]


def get_grouped_mm_configs(
    dtype_AB,
    exhaustive: bool = False,
) -> list[GluonGroupedMMConfig]:
    """
    Returns the configuration set for the Gluon Grouped MM kernel. Sized to
    land in the same ballpark as the CuTeDSL grouped-gemm heuristic's config
    counts (torch/_inductor/heuristics/template/cutedsl.py): ~22 for DEFAULT,
    ~800 for EXHAUSTIVE.

    Args:
        dtype_AB: Data type for A and B matrices
        exhaustive: If True, use full search space. Otherwise use handpicked configs.

    Returns:
        List of GluonGroupedMMConfig objects
    """
    if exhaustive:
        block_combos = list(itertools.product([64, 128], [32, 64, 128, 256]))
        BLOCK_K_vals = [64, 128, 256]
        NUM_STORE_WARP_vals = [4, 8, 16]
        buffer_configs_per_combo = 15
    else:
        block_combos = [
            (64, 32),
            (64, 64),
            (64, 128),
            (64, 256),
            (128, 64),
            (128, 128),
            (128, 256),
        ]
        BLOCK_K_vals = [64]
        NUM_STORE_WARP_vals = [4, 8, 16]
        buffer_configs_per_combo = 1

    configs = []
    for (BLOCK_M, BLOCK_N), BLOCK_K, num_store_warps in itertools.product(
        block_combos,
        BLOCK_K_vals,
        NUM_STORE_WARP_vals,
    ):
        buffer_variants = compute_stage_variants_gluon(
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            dtype=dtype_AB,
            max_configs=buffer_configs_per_combo,
        )

        for num_load_buffers, num_acc_buffers in buffer_variants:
            configs.append(
                GluonGroupedMMConfig(
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                    NUM_LOAD_BUFFERS=num_load_buffers,
                    NUM_ACC_BUFFERS=num_acc_buffers,
                    NUM_STORE_WARPS=num_store_warps,
                )
            )

    return configs
