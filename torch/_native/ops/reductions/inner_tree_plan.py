from typing import NamedTuple


_WARP_SIZE = 32
# Persistent/loop batch-size boundary (kInnerTreeThreshold).
_K_INNER_TREE_THRESHOLD = 8192
# Tiny-N multirow boundary (kMultiRowMaxLoads).
_K_MULTIROW_MAX_LOADS = 8
# Looped-vs-split boundary on num_batches (kTwoKernelThreshold).
_K_TWO_KERNEL_THRESHOLD = 3
_K_TARGET_WARPS_PER_BLOCK = 8  # kTargetWarpsPerBlock


def _previous_power_of_2(n: int) -> int:
    power = 1
    while power * 2 <= n:
        power <<= 1
    return power


def _next_power_of_2(n: int) -> int:
    power = 1
    while power < n:
        power <<= 1
    return power


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def vec_size(itemsize: int) -> int:
    """16-byte vectorized loads: fp32->4, fp64->2, fp16/bf16->8."""
    return max(1, 16 // max(1, itemsize))


class InnerTreeParams(NamedTuple):
    num_warps: int
    batch_total_elements: int  # Inductor R0_BLOCK (the per-batch tile).
    # Inductor split count: ceil(N / batch_total_elements).
    num_batches: int
    depth: int
    rows_per_block: int
    effective_loads: int


def compute_inner_tree_params(
    inputs_per_output: int, num_outputs: int, vec_size: int
) -> InnerTreeParams:
    """Plan the inner-tree reduction for one row of inputs.

    ``vec_size`` is ``vec_size(itemsize)``. The result is the tiling used by
    the eager kernel. Inductor consumes ``batch_total_elements`` (R0_BLOCK)
    and ``num_batches`` (split count).
    """
    wle = _WARP_SIZE * vec_size
    threshold = _K_INNER_TREE_THRESHOLD
    n = inputs_per_output

    if n > threshold:
        num_batches_est = _ceil_div(n, threshold)
        if num_batches_est <= _K_TWO_KERNEL_THRESHOLD:
            total_ideal_warps = 0
            for b in range(num_batches_est):
                batch_start = b * threshold
                batch_elements = min(threshold, n - batch_start)
                total_ideal_warps += min(16, max(1, batch_elements // wle))
            num_warps = max(1, total_ideal_warps // num_batches_est)
        else:
            num_warps = min(16, max(1, threshold // wle))
    else:
        num_warps = min(16, max(1, n // wle))
    num_warps = _previous_power_of_2(num_warps)

    loads_per_warp = _ceil_div(n, num_warps * wle)
    if loads_per_warp > 1:
        loads_per_warp = _next_power_of_2(loads_per_warp)

    max_loads_per_batch = max(1, threshold // (wle * num_warps))
    max_loads_per_batch = _previous_power_of_2(max_loads_per_batch)
    effective_loads = min(loads_per_warp, max_loads_per_batch)

    batch_total_elements = effective_loads * wle * num_warps
    num_batches = _ceil_div(n, batch_total_elements)

    if num_warps < _K_TARGET_WARPS_PER_BLOCK and num_batches <= _K_TWO_KERNEL_THRESHOLD:
        rows_per_block = min(num_outputs, _K_TARGET_WARPS_PER_BLOCK // num_warps)
    else:
        rows_per_block = 1

    depth = 0
    nn = effective_loads + 1
    while nn > 1:
        depth += 1
        nn >>= 1
    if depth < 1:
        depth = 1

    return InnerTreeParams(
        num_warps,
        batch_total_elements,
        num_batches,
        depth,
        rows_per_block,
        effective_loads,
    )
