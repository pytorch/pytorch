"""Kernel choice for the narrow out= sort, shared by the JIT router and the AOT
declaration (which loads this file by path). Torch-free: torchgen loads the
AOT declaration before torch is built.

The mirror kernels replace stock ``sort_cuda_kernel`` leaves at its thresholds:

  nsort <= 4096                         in-place leaves (Sort.cu); unstable
                                        rows of at most 32 take the bitonic one
  nsegments == 1 or nsort >= 1_000_000  one device radix sort per row
  nsegments < 128                       full sort with a segment payload
  otherwise                             segmented radix sort, one CTA per row
"""

INPLACE_MAX_NSORT = 4096
BITONIC_MAX_NSORT = 32
WARP_MERGE_MAX_NSORT = 128
PER_ROW_MIN_NSORT = 1_000_000
SEGMENTED_MIN_ROWS = 128
# The DSL kernels take the row count and the row length as Int32.
MAX_ROWS = (1 << 31) - 1
MAX_NSORT = (1 << 31) - 1
# Longest row the register kernel sorts; not a stock threshold.
REGISTER_MAX_NSORT = 2048
# Rows (warps) per CTA of the register kernel. On sm_120 two rows are best or
# within noise of best for every rung; one row loses up to 1.5x at rungs 32 and
# 64, and 16 rows spill 64-bit keys at rung 2048.
REGISTER_ROWS_PER_CTA = 2
# Longest bool row the partition kernel sorts: its flat extents are Int32, with
# room for one tile past the row end (cutedsl_partition.policy keeps a tile
# within 2^12 chunks of at most 16 bytes).
PARTITION_MAX_NSORT = (1 << 31) - (1 << 17)
INDEX_MAX = {
    "uint8": (1 << 8) - 1,
    "int8": (1 << 7) - 1,
    "uint16": (1 << 16) - 1,
    "int16": (1 << 15) - 1,
    "uint32": (1 << 32) - 1,
    "int32": (1 << 31) - 1,
    "int64": (1 << 63) - 1,
}


def inplace_sort_size(nsort: int, stable: bool) -> int:
    """Padded sort size of the in-place leaf route() sends a row of ``nsort``
    to: the bitonic leaf for unstable rows of at most BITONIC_MAX_NSORT, else
    the radix leaf (rows of REGISTER_MAX_NSORT + 1 .. INPLACE_MAX_NSORT)."""
    if not stable and nsort <= BITONIC_MAX_NSORT:
        return BITONIC_MAX_NSORT
    return INPLACE_MAX_NSORT


def register_rung(nsort: int) -> int:
    """Padded row length of the register kernel for rows of ``nsort``."""
    return max(32, 1 << (nsort - 1).bit_length())


def vector_alignment(store: int, itemsize: int) -> int:
    """Byte alignment of vectors of ``store`` elements of ``itemsize`` bytes."""
    return min(16, store * itemsize)


def register_store(nsort: int, operands: list[tuple[int, int]]) -> int:
    """Vector width of the register kernel: the widest power of two up to
    rung / 32 that divides the row and to which every ``(address, itemsize)``
    operand is aligned."""
    store = register_rung(nsort) // 32
    while store > 1 and (
        nsort % store
        or any(address % vector_alignment(store, size) for address, size in operands)
    ):
        store //= 2
    return store


def stock_leaf(nsort: int, nsegments: int) -> str:
    """Name of the stock CUDA sort leaf for a batch of ``nsegments`` rows."""
    if nsort <= INPLACE_MAX_NSORT:
        return "inplace"
    if nsegments == 1 or nsort >= PER_ROW_MIN_NSORT:
        return "per_row"
    if nsegments < SEGMENTED_MIN_ROWS:
        return "full_sort"
    return "segmented"


# Small calls whose JIT route launches several kernels from Python against one
# stock kernel lose to stock: the route's host cost (about 18 us per call on
# sm_120) exceeds the whole stock sort. Measured up to 2^16 elements for bool
# keys on the in-place leaf (partition: 1.16-1.22x stock) and single long rows
# (onesweep passes: up to 1.23x); every other measured short call is within 8%
# of stock or faster.
JIT_MIN_NUMEL = 1 << 16


def jit_pays_off(numel: int, nsort: int, nsegments: int, is_bool: bool) -> bool:
    """Whether the JIT route is worth its host cost for this call."""
    if numel > JIT_MIN_NUMEL:
        return True
    leaf = stock_leaf(nsort, nsegments)
    return not (leaf == "per_row" or (is_bool and leaf == "inplace"))


def route(nsort: int, nsegments: int, stable: bool, is_bool: bool) -> str:
    """Kernel for ``nsegments`` rows of ``nsort``: "partition", "register",
    "inplace" or "global". The JIT router and the AOT declaration both ask."""
    if is_bool and nsort <= PARTITION_MAX_NSORT:
        return "partition"
    if nsort <= REGISTER_MAX_NSORT and (stable or nsort > BITONIC_MAX_NSORT):
        return "register"
    return "inplace" if stock_leaf(nsort, nsegments) == "inplace" else "global"


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
