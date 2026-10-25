"""Leaf choice for the narrow out= sort kernels, shared by the JIT condition and
the AOT declaration. Torch-free: torchgen imports the AOT declaration before
torch is built.

The thresholds are those of the stock ``sort_cuda_kernel`` so that every DSL
kernel replaces exactly one stock leaf:

  nsort <= 4096                         in-place leaves (Sort.cu)
  nsegments == 1 or nsort >= 1_000_000  one device radix sort per row
  nsegments < 128                       full sort with a segment payload
  otherwise                             segmented radix sort, one CTA per row
"""

INPLACE_MAX_NSORT = 4096
PER_ROW_MIN_NSORT = 1_000_000
SEGMENTED_MIN_ROWS = 128
# The DSL kernels take the row count and the row length as Int32.
MAX_ROWS = (1 << 31) - 1
MAX_NSORT = (1 << 31) - 1
INDEX_MAX = {
    "uint8": (1 << 8) - 1,
    "int8": (1 << 7) - 1,
    "uint16": (1 << 16) - 1,
    "int16": (1 << 15) - 1,
    "uint32": (1 << 32) - 1,
    "int32": (1 << 31) - 1,
}


def stock_leaf(nsort: int, nsegments: int) -> str:
    """Name of the stock CUDA sort leaf for a batch of ``nsegments`` rows."""
    if nsort <= INPLACE_MAX_NSORT:
        return "inplace"
    if nsegments == 1 or nsort >= PER_ROW_MIN_NSORT:
        return "per_row"
    if nsegments < SEGMENTED_MIN_ROWS:
        return "full_sort"
    return "segmented"


# cub's tuning (cccl/cub/device/dispatch/tuning/tuning_radix_sort.cuh), resolved per
# arch like cub's ChainedPolicy. The stock sort passes an int64 item count, so cub
# picks 8-byte offsets (choose_offset_t) and the OFFSET_64BIT rows apply.
MAX_SMEM_PER_BLOCK = 48 * 1024  # cub detail::max_smem_per_block

# (key bytes, value bytes, offset bytes) -> (threads, items); value bytes 0 = keys only.
_SM90_SMALL_KEY = {
    (1, 0, 4): (512, 19),
    (1, 0, 8): (512, 19),
    (2, 0, 4): (512, 19),
    (2, 0, 8): (512, 19),
    (1, 1, 4): (512, 15),
    (1, 1, 8): (448, 16),
    (1, 2, 4): (512, 17),
    (1, 2, 8): (512, 14),
    (1, 4, 4): (512, 17),
    (1, 4, 8): (512, 14),
    (2, 1, 4): (384, 14),
    (2, 1, 8): (384, 16),
    (2, 2, 4): (384, 15),
    (2, 2, 8): (448, 16),
    (2, 4, 4): (512, 17),
    (2, 4, 8): (512, 12),
}
_SM100_SMALL_KEY = {
    **_SM90_SMALL_KEY,
    (2, 0, 4): (512, 20),
    (2, 0, 8): (384, 20),
    (1, 2, 4): (512, 18),
    (1, 4, 4): (512, 18),
    (1, 1, 8): (384, 20),
    (1, 2, 8): (256, 22),
    (1, 4, 8): (384, 15),
    (2, 1, 4): (448, 20),
    (2, 2, 4): (384, 23),
    (2, 1, 8): (384, 15),
    (2, 2, 8): (576, 15),
    (2, 4, 8): (512, 18),
}


def reg_bound_scaling(threads: int, items: int, dominant_bytes: int) -> tuple[int, int]:
    """cub's RegBoundScaling (util_arch.cuh) of a nominal 4-byte policy."""
    items = max(1, items * 4 // max(4, dominant_bytes))
    cap = -(-(MAX_SMEM_PER_BLOCK // (dominant_bytes * items)) // 32) * 32
    return min(threads, cap), items


def onesweep_policy(
    arch: int,
    key_bytes: int,
    value_bytes: int,
    offset_bytes: int = 8,
    float_keys: bool = False,
) -> tuple[int, int]:
    """cub's onesweep (threads, items) for ``arch`` = 10 * major + minor;
    ``value_bytes`` 0 means keys only."""
    offset64 = offset_bytes == 8
    keys_only = value_bytes == 0
    if arch >= 90:
        if key_bytes < 4:
            table = _SM100_SMALL_KEY if arch >= 100 else _SM90_SMALL_KEY
            nominal = table.get((key_bytes, value_bytes, offset_bytes), (384, 23))
        elif key_bytes == 4:  # OnesweepPolicyKey32
            if keys_only:
                nominal = (384, 20 - offset64 - float_keys)
            else:
                nominal = (
                    384,
                    (17 if offset64 else 23)
                    if value_bytes < 8
                    else (29 if offset64 else 30),
                )
        else:  # OnesweepPolicyKey64
            nominal = (384, 30 if value_bytes < 8 else 24)
    elif arch >= 80:
        nominal = (384, 17 if offset64 and key_bytes == 4 and not keys_only else 21)
    elif arch >= 70:
        nominal = (256, 46 if key_bytes == 4 and value_bytes == 4 else 23)
    else:
        nominal = (256, 30)
    return reg_bound_scaling(*nominal, max(key_bytes, value_bytes))


def histogram_policy(arch: int) -> tuple[int, int]:
    """cub's AgentRadixSortHistogramPolicy (threads, items)."""
    return (128, 16) if arch >= 80 else (256, 8)


def segmented_policies(
    arch: int, key_bytes: int, index_bytes: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """cub's (SegmentedPolicy, AltSegmentedPolicy) as (threads, items, radix bits)."""
    bits = 6 if key_bytes > 1 else 5
    if arch >= 60:
        nominal, alt = (192, 39), (384, 11)
    else:
        nominal, alt = (192, 31), (256, 11)
    dominant = max(key_bytes, index_bytes)
    return (
        (*reg_bound_scaling(*nominal, dominant), bits),
        (*reg_bound_scaling(*alt, dominant), bits - 1),
    )
