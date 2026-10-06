import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GluonGroupedMMConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    NUM_LOAD_BUFFERS: int
    NUM_ACC_BUFFERS: int
    NUM_STORE_WARPS: int = 4
    GROUP_SIZE_N: int = 1
    USE_TMA_STORE: bool = False
    USE_TWO_CTA: bool = False


def compute_stage_variants_gluon(
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype,
    tmem_max_columns: int = 512,
    max_configs: int = 1,
    uses_c_smem: bool = True,
    use_two_cta: bool = False,
):
    """
    Compute valid (NUM_LOAD_BUFFERS, NUM_ACC_BUFFERS) pairs for the
    given block dimensions, sampled evenly across the valid range so
    the result isn't biased toward the largest NUM_LOAD_BUFFERS that
    fits. Returns at most max_configs pairs.

    uses_c_smem must match the kernel: the C staging buffer only
    exists on the TMA store path, so budgeting for it when C is 2D
    reserves memory the kernel never uses and caps NUM_LOAD_BUFFERS.
    """
    import torch

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    smem_limit = 227 * 1024  # hardware limit

    a_bytes_per_stage = BLOCK_M * BLOCK_K * dtype_bytes
    b_bytes_per_stage = BLOCK_N * BLOCK_K * dtype_bytes
    if use_two_cta:
        b_bytes_per_stage //= 2
    c_bytes_per_stage = BLOCK_M * BLOCK_N * dtype_bytes if uses_c_smem else 0
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

    # Pipeline depth dominates: one CTA per SM leaves no spare warps to
    # hide load latency. Bound it by shared memory, not a constant.
    max_load_buffers = (smem_limit - c_bytes_per_stage) // (ab_bytes_per_stage + 8 * 2)

    all_valid = []
    for num_load_buffers in range(max_load_buffers, 0, -1):
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
    k_is_varying: bool = False,
) -> list[GluonGroupedMMConfig]:
    """
    Returns the configuration set for the Gluon Grouped MM kernel,
    sized to match the CuTeDSL grouped-gemm heuristic's config counts
    (torch/_inductor/heuristics/template/cutedsl.py).

    Args:
        dtype_AB: Data type for A and B matrices
        k_is_varying: Whether offs partitions K

    Returns:
        List of GluonGroupedMMConfig objects
    """
    # NUM_STORE_WARPS measured under 1% across 4/8/16, and the deepest
    # pipeline that fits won on every shape tried, so neither earns a
    # search dimension. BLOCK_K=32 only wins when offs partitions K,
    # where a group's K can be far smaller than the tile. The store is
    # searched both ways: TMA needs a C staging buffer, which costs a
    # pipeline stage, so which wins is shape-dependent.
    #
    # No EXHAUSTIVE variant: a wider space (more BLOCK_K/BLOCK_M/
    # BLOCK_N/GROUP_SIZE_N values) measured no improvement over this.
    block_combos = [
        (64, 32),
        (64, 64),
        (64, 128),
        (64, 256),
        (128, 64),
        (128, 128),
        (128, 256),
    ]
    BLOCK_K_vals = [32, 64, 128] if k_is_varying else [64, 128]
    NUM_STORE_WARP_vals = [8]
    GROUP_SIZE_N_vals = [1, 8]
    USE_TMA_STORE_vals = [False, True]
    USE_TWO_CTA_vals = [False, True]
    buffer_configs_per_combo = 1

    configs = []
    for (
        (BLOCK_M, BLOCK_N),
        BLOCK_K,
        num_store_warps,
        group_size_n,
        use_tma_store,
        use_two_cta,
    ) in itertools.product(
        block_combos,
        BLOCK_K_vals,
        NUM_STORE_WARP_vals,
        GROUP_SIZE_N_vals,
        USE_TMA_STORE_vals,
        USE_TWO_CTA_vals,
    ):
        buffer_variants = compute_stage_variants_gluon(
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            dtype=dtype_AB,
            max_configs=buffer_configs_per_combo,
            uses_c_smem=use_tma_store,
            use_two_cta=use_two_cta,
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
                    GROUP_SIZE_N=group_size_n,
                    USE_TMA_STORE=use_tma_store,
                    USE_TWO_CTA=use_two_cta,
                )
            )

    return configs


def prune_grouped_mm_configs(
    configs: list[GluonGroupedMMConfig],
    G: int,
    M: int,
    N: int,
    K: int,
    a_is_2d: bool,
    b_is_2d: bool,
    num_sms: int,
) -> list[GluonGroupedMMConfig]:
    # Average per-group extents; the ragged dimension is split across G.
    m_g = M / G if a_is_2d and not b_is_2d else M
    n_g = N / G if b_is_2d and not a_is_2d else N
    k_g = K / G if a_is_2d and b_is_2d else K

    min_m = min(c.BLOCK_M for c in configs)
    # Padded N tiles up to 128 still win on tiny N, so they are never pruned.
    min_n = max(min(c.BLOCK_N for c in configs), 128)
    min_k = min(c.BLOCK_K for c in configs)
    many_waves = (
        m_g >= 128
        and n_g >= 128
        and (G * math.ceil(m_g / 128) * math.ceil(n_g / 256) >= 4 * num_sms)
    )

    def keep(c: GluonGroupedMMConfig) -> bool:
        tile_m = c.BLOCK_M * (2 if c.USE_TWO_CTA else 1)
        # A tile covering the extent even at half its size only adds padding.
        if tile_m > min_m and m_g <= tile_m / 2:
            return False
        if c.BLOCK_N > min_n and n_g <= c.BLOCK_N / 2:
            return False
        if c.BLOCK_K > min_k and k_g <= c.BLOCK_K / 2:
            return False
        # With several full waves of work, small tiles only lose data reuse.
        return not (many_waves and c.BLOCK_M * c.BLOCK_N < 128 * 128)

    return [c for c in configs if keep(c)] or configs
