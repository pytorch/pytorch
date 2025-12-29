from dataclasses import dataclass
import itertools

import torch._inductor.config as config


@dataclass(frozen=True)
class GluonGroupedMMConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    NUM_LOAD_BUFFERS: int
    NUM_ACC_BUFFERS: int
    NUM_LOAD_WARPS: int = 1
    NUM_COMPUTE_WARPS: int = 1
    NUM_STORE_WARPS: int = 4
    NUM_LOAD_THREAD_REGISTERS: int = 24
    NUM_COMPUTE_THREAD_REGISTERS: int = 24
    MAXNREG: int = 128


def compute_stage_variants_gluon(
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype,
    num_store_warps: int = 4,
    occupancy: int = 1,
    smem_capacity: int = 228 * 1024,
    tmem_max_columns: int = 512,
    exhaustive: bool = False,
    max_configs: int = 4,
):
    """
    Compute valid buffer configurations for given block dimensions.
    Returns list of (num_load_buffers, num_acc_buffers) tuples.

    Args:
        exhaustive: If False, limit to max_configs best configurations
        max_configs: Maximum number of configs to return when not exhaustive
    """
    import torch

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    smem_limit = 232448  # 227 KB hardware limit

    # Calculate SMEM usage
    a_bytes_per_stage = BLOCK_M * BLOCK_K * dtype_bytes
    b_bytes_per_stage = BLOCK_N * BLOCK_K * dtype_bytes
    c_bytes_per_stage = BLOCK_M * BLOCK_N * dtype_bytes
    ab_bytes_per_stage = a_bytes_per_stage + b_bytes_per_stage

    # Check minimum config fits
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

    valid_configs = []

    # Try all combinations of load/acc buffers
    for num_load_buffers in range(8, 0, -1):
        ab_smem = ab_bytes_per_stage * num_load_buffers
        c_smem = c_bytes_per_stage
        load_barrier_smem = 8 * num_load_buffers * 2

        base_smem = ab_smem + c_smem + load_barrier_smem + compiler_overhead

        if base_smem > smem_limit:
            continue

        # Try different acc buffer counts
        max_acc_by_tmem = tmem_max_columns // BLOCK_N
        remaining_smem = smem_limit - base_smem
        max_acc_by_smem = remaining_smem // (8 * 2)

        max_acc_buffers = min(max_acc_by_tmem, max_acc_by_smem, 8)

        for num_acc_buffers in range(max_acc_buffers, 0, -1):
            acc_barrier_smem = 8 * num_acc_buffers * 2
            total_smem = base_smem + acc_barrier_smem
            tmem_cols = BLOCK_N * num_acc_buffers

            if total_smem <= smem_limit and tmem_cols <= tmem_max_columns:
                valid_configs.append((num_load_buffers, num_acc_buffers))

                # Limit configs for non-exhaustive search
                if not exhaustive and len(valid_configs) >= max_configs:
                    return valid_configs

    return valid_configs


def get_grouped_mm_configs(
    dtype_AB,
    dtype_C=None,
    dtype_acc=None,
    M=None,
    N=None,
    K=None,
    exhaustive: bool = False
) -> list[GluonGroupedMMConfig]:
    """
    Returns the configuration set for the Gluon Grouped MM kernel.

    Args:
        dtype_AB: Data type for A and B matrices
        dtype_C: Data type for C matrix (unused, for compatibility)
        dtype_acc: Data type for accumulation (unused, for compatibility)
        M, N, K: Problem dimensions (optional)
        exhaustive: If True, use full search space. Otherwise use handpicked configs.

    Returns:
        List of GluonGroupedMMConfig objects
    """
    configs = []

    if exhaustive:
        # Full ranges for exhaustive search
        BLOCK_M_vals = [64, 128]
        BLOCK_N_vals = [64, 128, 256]
        BLOCK_K_vals = [64, 128, 256]
        NUM_LOAD_WARP_vals = [1, 2]
        NUM_COMPUTE_WARP_vals = [1, 2]
        NUM_STORE_WARP_vals = [4, 8]
    else:
        # Default configs based on CuTeDSL pattern:
        # Extract BLOCK_M and BLOCK_N from CuTeDSL's default configs where CLUSTER=(1,1)
        # Note: Gluon TMA requires power-of-2 block shapes, so we filter out 160, 192
        # CuTeDSL configs with CLUSTER_M=1, CLUSTER_N=1 (power-of-2 only):
        # (64, 32), (64, 64), (64, 256), (128, 64), (128, 256)
        block_mn_pairs = [
            (64, 32),
            (64, 64),
            (64, 128),
            (64, 256),
            (128, 64),
            (128, 128),
            (128, 256),
        ]
        # For Gluon, BLOCK_K seems less critical - use fixed value
        BLOCK_K_vals = [64]
        # Use "cluster (1,1)" equivalent: NUM_LOAD_WARPS=1, NUM_COMPUTE_WARPS=1
        NUM_LOAD_WARP_vals = [1]
        NUM_COMPUTE_WARP_vals = [1]
        NUM_STORE_WARP_vals = [4, 8]

    NUM_LOAD_THREAD_REGISTERS_vals = [24]
    NUM_COMPUTE_THREAD_REGISTERS_vals = [24]
    MAXNREG_vals = [128]

    if exhaustive:
        # Exhaustive: iterate over all combinations
        for (
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_load_warps,
            num_compute_warps,
            num_store_warps,
            num_load_thread_registers,
            num_compute_thread_registers,
            maxnreg,
        ) in itertools.product(
            BLOCK_M_vals,
            BLOCK_N_vals,
            BLOCK_K_vals,
            NUM_LOAD_WARP_vals,
            NUM_COMPUTE_WARP_vals,
            NUM_STORE_WARP_vals,
            NUM_LOAD_THREAD_REGISTERS_vals,
            NUM_COMPUTE_THREAD_REGISTERS_vals,
            MAXNREG_vals,
        ):
            buffer_variants = compute_stage_variants_gluon(
                BLOCK_M, BLOCK_N, BLOCK_K, dtype=dtype_AB, num_store_warps=num_store_warps, exhaustive=True
            )

            for num_load_buffers, num_acc_buffers in buffer_variants:
                total_regs = (
                    (num_load_warps + num_compute_warps + num_store_warps) * 32 * maxnreg
                )
                REGS_PER_SM = 65536
                MAX_CTAS_PER_SM = 32
                estimated_occupancy = min(REGS_PER_SM // total_regs, MAX_CTAS_PER_SM)

                if estimated_occupancy < 1:
                    continue

                configs.append(
                    GluonGroupedMMConfig(
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                        NUM_LOAD_BUFFERS=num_load_buffers,
                        NUM_ACC_BUFFERS=num_acc_buffers,
                        NUM_LOAD_WARPS=num_load_warps,
                        NUM_COMPUTE_WARPS=num_compute_warps,
                        NUM_STORE_WARPS=num_store_warps,
                        NUM_LOAD_THREAD_REGISTERS=num_load_thread_registers,
                        NUM_COMPUTE_THREAD_REGISTERS=num_compute_thread_registers,
                        MAXNREG=maxnreg,
                    )
                )
    else:
        # Default: use handpicked (BLOCK_M, BLOCK_N) pairs with cluster (1,1)
        for (
            (BLOCK_M, BLOCK_N),
            BLOCK_K,
            num_load_warps,
            num_compute_warps,
            num_store_warps,
            num_load_thread_registers,
            num_compute_thread_registers,
            maxnreg,
        ) in itertools.product(
            block_mn_pairs,
            BLOCK_K_vals,
            NUM_LOAD_WARP_vals,
            NUM_COMPUTE_WARP_vals,
            NUM_STORE_WARP_vals,
            NUM_LOAD_THREAD_REGISTERS_vals,
            NUM_COMPUTE_THREAD_REGISTERS_vals,
            MAXNREG_vals,
        ):
            buffer_variants = compute_stage_variants_gluon(
                BLOCK_M, BLOCK_N, BLOCK_K, dtype=dtype_AB, num_store_warps=num_store_warps, exhaustive=False
            )

            for num_load_buffers, num_acc_buffers in buffer_variants:
                total_regs = (
                    (num_load_warps + num_compute_warps + num_store_warps) * 32 * maxnreg
                )
                REGS_PER_SM = 65536
                MAX_CTAS_PER_SM = 32
                estimated_occupancy = min(REGS_PER_SM // total_regs, MAX_CTAS_PER_SM)

                if estimated_occupancy < 1:
                    continue

                configs.append(
                    GluonGroupedMMConfig(
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        BLOCK_K=BLOCK_K,
                        NUM_LOAD_BUFFERS=num_load_buffers,
                        NUM_ACC_BUFFERS=num_acc_buffers,
                        NUM_LOAD_WARPS=num_load_warps,
                        NUM_COMPUTE_WARPS=num_compute_warps,
                        NUM_STORE_WARPS=num_store_warps,
                        NUM_LOAD_THREAD_REGISTERS=num_load_thread_registers,
                        NUM_COMPUTE_THREAD_REGISTERS=num_compute_thread_registers,
                        MAXNREG=maxnreg,
                    )
                )

    return configs
