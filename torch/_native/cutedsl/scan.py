"""Warp- and block-wide prefix scans shared by native CuTeDSL operators."""

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32


@cute.jit
def warp_inclusive_prefix(val: Int32, lane: Int32) -> Int32:
    """Inclusive Hillis-Steele prefix sum across a single warp."""
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # mask_and_clamp=0 so lanes < offset get 0 (no contribution).
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= Int32(offset):
            val = val + partial
    return val


@cute.jit
def block_exclusive_prefix(
    val: Int32,
    scratch: cute.Tensor,
    lane: Int32,
    warp: Int32,
    num_warps: cutlass.Constexpr,
) -> tuple[Int32, Int32]:
    """Block-wide exclusive prefix sum over per-thread ``val``.

    Returns ``(excl_pfx, total)``: ``excl_pfx`` is this thread's
    exclusive prefix; ``total`` is the sum across all threads in the
    block. ``scratch`` must have at least ``num_warps + 1`` Int32 slots
    and is clobbered.

    Implementation: warp-level Hillis-Steele scan, then warp 0 scans
    the per-warp totals to produce per-warp offsets, then add.
    """
    WARP_SIZE = const_expr(cute.arch.WARP_SIZE)
    # Step 1: per-warp inclusive scan.
    incl = warp_inclusive_prefix(val, lane)
    excl_intra = incl - val
    # Step 2: last lane in each warp publishes warp's inclusive total.
    if lane == Int32(WARP_SIZE - 1):
        scratch[warp] = incl
    cute.arch.barrier()
    # Step 3: warp 0 prefix-sums the per-warp totals.
    if warp == Int32(0):
        warp_val = Int32(0)
        if lane < Int32(num_warps):
            warp_val = scratch[lane]
        warp_incl = warp_inclusive_prefix(warp_val, lane)
        warp_excl = warp_incl - warp_val
        if lane < Int32(num_warps):
            scratch[lane] = warp_excl
        # Stash the block total in a slot beyond the per-warp set so
        # other warps can read it without a second reduction.
        if lane == Int32(num_warps - 1):
            scratch[num_warps] = warp_incl
    cute.arch.barrier()
    excl_pfx = scratch[warp] + excl_intra
    total = scratch[num_warps]
    cute.arch.barrier()  # let everyone consume scratch before next call
    return excl_pfx, total
