"""CuTeDSL fp32 top-K kernels.

Two kernels live here, picked by K and N in ``cutedsl_impl.py``:

``topk_radix`` - fused radix-select for K in {64,128,256,512,1024}.
Layout per CTA (one row):
  * Phase 1 - four radix byte passes (MSB to LSB) build shared-memory
    histograms via smem atomic-add; thread 0 scans descending to locate
    the bin that contains the k-th-largest ordinal.
  * Phase 2 - gather elements at-or-above the radix threshold. Two
    specialisations selected by the ``deterministic`` constexpr:
      - True: per-step block-wide exclusive prefix sums hand each
        thread an exact write slot, eq-tied elements stage in a side
        buffer; first ``remaining_k`` of them are appended in input
        order. Stable across runs.
      - False: smem atomic counters claim slots on the fly. Cheaper
        but indices on threshold ties depend on atomic resolution
        order.
  * Phase 3 - cooperative bitonic sort descending. Block size is
    ``max(K, 256)``; K threads own one element each, the rest sit the
    sort out but still reach every barrier. Comparator is lex
    ``(ord, -idx)`` (deterministic) or ord-only (non-deterministic).
  * Phase 4 - write (values, indices) to global memory.

``topk_register`` - register-resident top-K for K in {16, 32} with
N a power of 2 in [K, 2048]. Each warp owns one row; ``ROWS_PER_CTA``
warps share a CTA. Per-thread VEC = N/32 elements held in registers,
encoded as Int64 ``(ord << 32) | ~idx`` keys so a descending sort gives
``(value desc, idx asc)`` matching aten. Local bitonic sort, then a
warp-cooperative bitonic-topk merge across butterfly partners. No smem,
no global staging - only the final K writes hit gmem. Bit-exact to aten.

Both kernels do one launch; only loads/stores of the input and final
outputs hit global memory.
"""

import math

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch._vendor.quack.cache import jit_cache


_NEG_INF_BITS: int = 0xFF800000
_N_HIST_BINS: int = 256


def _num_threads_for_k(k: int) -> int:
    """Pick block size for a given K.

    Bitonic sort in phase 3 assigns one element per thread, so we need
    at least K threads. Using ``max(K, _N_HIST_BINS)`` keeps the phase-1
    histogram-zero loop one-shot for small K while letting K grow up to
    the CUDA 1024-thread cap.
    """
    return max(k, _N_HIST_BINS)


@dsl_user_op
def _elem_ptr(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def _bitcast_f32_to_i32(val, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(val).ir_value()))


@dsl_user_op
def _bitcast_i32_to_f32(val, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(val).ir_value()))


@cute.jit
def _f32_to_radix_ord(val: Float32) -> Int32:
    """Map fp32 to a monotone Int32 ordinal whose unsigned MSB->LSB byte
    order matches descending fp32 order. Negatives invert all bits;
    non-negatives keep their bits. With the byte-0 xor of 0x80 (applied
    at the histogram-bin level), this gives a total order suitable for
    MSB-first radix select.

    NaN of either sign maps to 0x7FFFFFFF (the largest ordinal under the
    byte-0 xor), matching aten's TopKTypeConfig<float> behaviour: NaN
    sorts at the very top of largest-first topk regardless of sign.
    """
    bits = _bitcast_f32_to_i32(val)
    ord_ = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    # NaN detection: bits with all-ones exponent and non-zero mantissa.
    # Equivalent to ``val != val``, but the DSL doesn't expose that
    # comparator; the bit test is exact for IEEE-754 fp32.
    is_nan = (bits & Int32(0x7FFFFFFF)) > Int32(0x7F800000)
    return Int32(0x7FFFFFFF) if is_nan else ord_


def _smem_scalar(smem: cutlass.utils.SmemAllocator, dtype) -> cute.Tensor:
    """Allocate a single-element smem tensor (scalar bookkeeping slot)."""
    return smem.allocate_tensor(dtype, cute.make_layout((1,)), byte_alignment=4)


@cute.jit
def _load_vec_f32(ptr: cute.Pointer, vec_elems: cutlass.Constexpr) -> cute.Tensor:
    """Vectorized gmem -> rmem load of ``vec_elems`` contiguous fp32.

    Cute picks the widest safe instruction (e.g. LDG.128 for a 4-wide
    fp32 tile) from the static shape.
    """
    gsrc = cute.make_tensor(ptr, cute.make_layout(vec_elems))
    rvals = cute.make_rmem_tensor(vec_elems, Float32)
    cute.autovec_copy(gsrc, rvals)
    return rvals


@cute.jit
def _warp_inclusive_prefix(val: Int32, lane: Int32) -> Int32:
    """Inclusive Hillis-Steele prefix sum across a single warp."""
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # mask_and_clamp=0 so lanes < offset get 0 (no contribution).
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= Int32(offset):
            val = val + partial
    return val


@cute.jit
def _block_excl_prefix_i32(
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
    incl = _warp_inclusive_prefix(val, lane)
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
        warp_incl = _warp_inclusive_prefix(warp_val, lane)
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


class _RadixSelectTopK:
    """One-CTA-per-row fused radix-select + bitonic sort.

    ``N``, ``K`` and ``deterministic`` are specialised at compile time:
    the JIT cache keys on ``(N, K, deterministic)`` so a row-length or
    determinism-mode change forces a recompile.

    When ``deterministic`` is True, phase 2 uses block-wide prefix-sum
    scans for input-stable gather and phase 3's bitonic comparator uses
    a lex ``(ord, -idx)`` key. Output matches aten bit-exactly.

    When ``deterministic`` is False, phase 2 uses smem atomic counters
    for the gather and phase 3 sorts by ord-only. Faster (no extra smem
    bookkeeping, no per-step block barrier on the gather), but ties at
    the threshold radix bin are resolved non-deterministically and
    indices may differ across runs.
    """

    VEC: int = 4

    def __init__(self, N: int, K: int, deterministic: bool = True):
        self.N = N
        self.K = K
        self.deterministic = deterministic
        self.NUM_THREADS = _num_threads_for_k(K)
        self.NUM_STAGES = int(math.log2(K))
        TILE = self.NUM_THREADS * self.VEC
        self.VEC_ITERS = N // TILE
        self.VEC_TAIL_START = self.VEC_ITERS * TILE
        self.SCALAR_TAIL_ITERS = (
            N - self.VEC_TAIL_START + self.NUM_THREADS - 1
        ) // self.NUM_THREADS

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M = mX.shape[0]
        self.kernel(mX, mValues, mIndices).launch(
            grid=[M, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        N = const_expr(self.N)
        K = const_expr(self.K)
        NT = const_expr(self.NUM_THREADS)
        VEC = const_expr(self.VEC)
        VEC_ITERS = const_expr(self.VEC_ITERS)
        VEC_TAIL_START = const_expr(self.VEC_TAIL_START)
        SCALAR_TAIL_ITERS = const_expr(self.SCALAR_TAIL_ITERS)
        N_HIST_BINS = const_expr(_N_HIST_BINS)

        smem = cutlass.utils.SmemAllocator()
        s_hist = smem.allocate_tensor(
            Int32, cute.make_layout((256,)), byte_alignment=16
        )
        s_prefix = _smem_scalar(smem, Int32)
        s_mask = _smem_scalar(smem, Int32)
        s_rem_k = _smem_scalar(smem, Int32)
        # Phase-2 bookkeeping. Non-deterministic gather uses two
        # atomic-add counters; deterministic gather carries running
        # totals in registers.
        s_write_ctr = _smem_scalar(smem, Int32)
        s_eq_ctr = _smem_scalar(smem, Int32)
        s_vals = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        s_ords = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
        s_idxs = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
        # Side buffer for equal-threshold elements (deterministic gather only).
        # Always allocated to keep the smem layout straightforward;
        # measurements show no occupancy benefit from making it conditional.
        s_eq_vals = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        s_eq_idxs = smem.allocate_tensor(
            Int32, cute.make_layout((K,)), byte_alignment=16
        )

        if tidx == 0:
            s_prefix[0] = Int32(0)
            s_mask[0] = Int32(0)
            s_rem_k[0] = Int32(K)
        cute.arch.barrier()

        # Phase 1: four radix byte passes, MSB to LSB.
        for byte_pos in cutlass.range_constexpr(4):
            shift = (3 - byte_pos) * 8
            xor_val = 128 if byte_pos == 0 else 0

            # Zero the 256-bin histogram. When NT > 256 (e.g. K=1024) only
            # the first 256 threads participate; the histogram is always
            # exactly 256 bins regardless of K.
            if tidx < N_HIST_BINS:
                s_hist[tidx] = Int32(0)
            cute.arch.barrier()

            prefix = s_prefix[0]
            decided_mask = s_mask[0]

            if byte_pos == 0:
                # First pass: decided_mask==0, every element participates.
                for step in cutlass.range(VEC_ITERS):
                    base = step * NT * VEC + tidx * VEC
                    rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                    for vi in cutlass.range_constexpr(self.VEC):
                        ords = _f32_to_radix_ord(rvals[vi])
                        byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                            xor_val
                        )
                        cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
                for step in cutlass.range(SCALAR_TAIL_ITERS):
                    idx = VEC_TAIL_START + step * NT + tidx
                    if idx < N:
                        ords = _f32_to_radix_ord(mX[row, idx])
                        byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                            xor_val
                        )
                        cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
            else:
                for step in cutlass.range(VEC_ITERS):
                    base = step * NT * VEC + tidx * VEC
                    rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                    for vi in cutlass.range_constexpr(self.VEC):
                        ords = _f32_to_radix_ord(rvals[vi])
                        if (ords & decided_mask) == prefix:
                            byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                                xor_val
                            )
                            cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
                for step in cutlass.range(SCALAR_TAIL_ITERS):
                    idx = VEC_TAIL_START + step * NT + tidx
                    if idx < N:
                        ords = _f32_to_radix_ord(mX[row, idx])
                        if (ords & decided_mask) == prefix:
                            byte_val = ((ords >> Int32(shift)) & Int32(255)) ^ Int32(
                                xor_val
                            )
                            cute.arch.atomic_add(_elem_ptr(s_hist, byte_val), Int32(1))
            cute.arch.barrier()

            # Thread 0 picks the bin that contains the k-th-largest.
            if tidx == 0:
                remaining_k = s_rem_k[0]
                acc = Int32(0)
                found = Int32(0)
                sel_bin = Int32(0)
                elems_above = Int32(0)

                for b in cutlass.range(256):
                    bin_idx = Int32(255) - b
                    count = s_hist[bin_idx]
                    if found == Int32(0):
                        if acc + count >= remaining_k:
                            sel_bin = bin_idx
                            elems_above = acc
                            found = Int32(1)
                    acc = acc + count

                actual_byte = sel_bin ^ Int32(xor_val)
                s_prefix[0] = prefix | (actual_byte << Int32(shift))
                s_mask[0] = decided_mask | (Int32(255) << Int32(shift))
                s_rem_k[0] = remaining_k - elems_above

            cute.arch.barrier()

        # Phase 2: gather elements at-or-above the radix threshold.
        # Two implementations selected at compile time:
        #   * Deterministic: per-step block-wide exclusive prefix sums
        #     give each thread an exact write slot; eq-threshold ties
        #     stage in a side buffer and the first ``remaining_k`` get
        #     appended in input-index order.
        #   * Non-deterministic: smem atomic counters claim slots on
        #     the fly. Faster but tied elements at the threshold bin
        #     are picked in atomic-resolution order.
        threshold = s_prefix[0]
        remaining_k = s_rem_k[0]

        if tidx == 0:
            s_write_ctr[0] = Int32(0)
            s_eq_ctr[0] = Int32(0)
        if tidx < K:
            s_vals[tidx] = _bitcast_i32_to_f32(Int32(_NEG_INF_BITS))
            s_idxs[tidx] = Int32(0)
        cute.arch.barrier()

        if const_expr(self.deterministic):
            WARP_SIZE = const_expr(cute.arch.WARP_SIZE)
            NUM_WARPS = const_expr(NT // WARP_SIZE)
            lane_idx = tidx % Int32(WARP_SIZE)
            warp_idx = tidx // Int32(WARP_SIZE)

            # --- Vec + scalar tail (deterministic) ---
            # Pack (n_above, n_eq) into a single Int32 as (high16, low16) so
            # we can run one block-wide prefix sum per step instead of two.
            # Per-step bounds: above+eq counts <= VEC*NT = 4*1024 = 4096,
            # comfortably below 2^16; the sums share no bits so the low-half
            # never carries into the high half.
            #
            # Carry the running bases in registers (every thread has
            # ``packed_step_total`` from the scan) instead of going through
            # smem each step.
            above_base = Int32(0)
            eq_base = Int32(0)
            for step in cutlass.range(VEC_ITERS):
                base = step * NT * VEC + tidx * VEC
                rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)

                # Hoist per-element classification into per-lane registers
                # so the count pass and write pass don't redo the radix-ord
                # transform and threshold compare.
                # Class encoding: 2 = above, 1 = eq, 0 = below.
                cls_reg = cute.make_rmem_tensor(VEC, Int32)
                packed_local = Int32(0)
                for vi in cutlass.range_constexpr(self.VEC):
                    o = _f32_to_radix_ord(rvals[vi])
                    if o > threshold:
                        cls_reg[vi] = Int32(2)
                        packed_local = packed_local + Int32(1 << 16)
                    elif o == threshold:
                        cls_reg[vi] = Int32(1)
                        packed_local = packed_local + Int32(1)
                    else:
                        cls_reg[vi] = Int32(0)

                # Block-wide exclusive prefix sum. ``s_hist`` is reused as
                # scratch for the warp-total scan (Phase 1 is done with it).
                packed_pfx, packed_step_total = _block_excl_prefix_i32(
                    packed_local, s_hist, lane_idx, warp_idx, NUM_WARPS
                )
                above_pfx = packed_pfx >> Int32(16)
                eq_pfx = packed_pfx & Int32(0xFFFF)

                # Per-thread write offsets within this step.
                my_above = above_base + above_pfx
                my_eq = eq_base + eq_pfx
                for vi in cutlass.range_constexpr(self.VEC):
                    val_f32 = rvals[vi]
                    cls = cls_reg[vi]
                    idx = base + Int32(vi)
                    if cls == Int32(2):
                        if my_above < Int32(K):
                            s_vals[my_above] = val_f32
                            s_idxs[my_above] = Int32(idx)
                        my_above = my_above + Int32(1)
                    elif cls == Int32(1):
                        if my_eq < remaining_k:
                            s_eq_vals[my_eq] = val_f32
                            s_eq_idxs[my_eq] = Int32(idx)
                        my_eq = my_eq + Int32(1)

                # Advance running bases. Cap eq at remaining_k - we never
                # need more. All threads update in registers; no smem.
                above_base = above_base + (packed_step_total >> Int32(16))
                eq_base = min(
                    remaining_k, eq_base + (packed_step_total & Int32(0xFFFF))
                )

            # --- Scalar tail ---
            for step in cutlass.range(SCALAR_TAIL_ITERS):
                idx = VEC_TAIL_START + step * NT + tidx
                valid = idx < N
                packed_local = Int32(0)
                if valid:
                    ords = _f32_to_radix_ord(mX[row, idx])
                    if ords > threshold:
                        packed_local = Int32(1 << 16)
                    elif ords == threshold:
                        packed_local = Int32(1)

                packed_pfx, packed_step_total = _block_excl_prefix_i32(
                    packed_local, s_hist, lane_idx, warp_idx, NUM_WARPS
                )
                above_pfx = packed_pfx >> Int32(16)
                eq_pfx = packed_pfx & Int32(0xFFFF)

                my_above = above_base + above_pfx
                my_eq = eq_base + eq_pfx
                if valid:
                    val_f32 = mX[row, idx]
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        if my_above < Int32(K):
                            s_vals[my_above] = val_f32
                            s_idxs[my_above] = Int32(idx)
                    elif ords == threshold:
                        if my_eq < remaining_k:
                            s_eq_vals[my_eq] = val_f32
                            s_eq_idxs[my_eq] = Int32(idx)

                above_base = above_base + (packed_step_total >> Int32(16))
                eq_base = min(
                    remaining_k, eq_base + (packed_step_total & Int32(0xFFFF))
                )

            # The eq-merge below has each thread read s_eq_vals[tidx - n_above],
            # i.e. a slot written by some other thread during the gather above.
            # The last barrier inside _block_excl_prefix_i32 fires *before* the
            # final per-step writes to s_eq_vals, so cross-thread visibility of
            # those writes isn't guaranteed without a fresh barrier here.
            cute.arch.barrier()

            # Append eq-bucket survivors after the above-bucket ones to
            # fill ``s_vals`` / ``s_idxs`` to exactly K entries.
            n_above = above_base
            if tidx < K:
                offset_into_eq = Int32(tidx) - n_above
                if offset_into_eq >= Int32(0) and offset_into_eq < remaining_k:
                    s_vals[tidx] = s_eq_vals[offset_into_eq]
                    s_idxs[tidx] = s_eq_idxs[offset_into_eq]
            cute.arch.barrier()
        else:
            # Non-deterministic atomic-counter gather.
            for step in cutlass.range(VEC_ITERS):
                base = step * NT * VEC + tidx * VEC
                rvals = _load_vec_f32(_elem_ptr(mX, (row, base)), VEC)
                for vi in cutlass.range_constexpr(self.VEC):
                    val_f32 = rvals[vi]
                    idx = base + Int32(vi)
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        pos = cute.arch.atomic_add(_elem_ptr(s_write_ctr, 0), Int32(1))
                        if pos < Int32(K):
                            s_vals[pos] = val_f32
                            s_idxs[pos] = Int32(idx)
                    elif ords == threshold:
                        eq_pos = cute.arch.atomic_add(_elem_ptr(s_eq_ctr, 0), Int32(1))
                        if eq_pos < remaining_k:
                            pos = cute.arch.atomic_add(
                                _elem_ptr(s_write_ctr, 0), Int32(1)
                            )
                            if pos < Int32(K):
                                s_vals[pos] = val_f32
                                s_idxs[pos] = Int32(idx)
            for step in cutlass.range(SCALAR_TAIL_ITERS):
                idx = VEC_TAIL_START + step * NT + tidx
                if idx < N:
                    val_f32 = mX[row, idx]
                    ords = _f32_to_radix_ord(val_f32)
                    if ords > threshold:
                        pos = cute.arch.atomic_add(_elem_ptr(s_write_ctr, 0), Int32(1))
                        if pos < Int32(K):
                            s_vals[pos] = val_f32
                            s_idxs[pos] = Int32(idx)
                    elif ords == threshold:
                        eq_pos = cute.arch.atomic_add(_elem_ptr(s_eq_ctr, 0), Int32(1))
                        if eq_pos < remaining_k:
                            pos = cute.arch.atomic_add(
                                _elem_ptr(s_write_ctr, 0), Int32(1)
                            )
                            if pos < Int32(K):
                                s_vals[pos] = val_f32
                                s_idxs[pos] = Int32(idx)
            cute.arch.barrier()

        # Phase 3: cooperative bitonic sort (descending) on a composite
        # ``(ord, ~idx)`` Int64 key so equal-value entries break ties by
        # input index, matching aten's stable ordering.
        # K lanes own one element each; lanes >= K clamp their smem
        # index to a dummy slot so they can participate in barriers
        # without OOB reads/writes. The writes from those lanes are
        # discarded via the `tidx < K` guard on the store.
        sort_tidx = tidx if tidx < Int32(K) else Int32(0)
        if tidx < K:
            s_ords[tidx] = _f32_to_radix_ord(s_vals[sort_tidx])
        cute.arch.barrier()

        for stage in cutlass.range_constexpr(self.NUM_STAGES):
            for sub_rev in cutlass.range_constexpr(stage + 1):
                sub = stage - sub_rev
                step_size = 1 << sub

                # Snapshot both participants before the barrier; after the
                # barrier the other thread may have overwritten its slot,
                # so we must read our own ord now. Inactive lanes clamp
                # their partner index to 0 (a safe in-bounds read) - their
                # resulting comparisons are discarded by the store guard.
                partner = (tidx ^ Int32(step_size)) if tidx < Int32(K) else Int32(0)
                my_o = s_ords[sort_tidx]
                my_i = s_idxs[sort_tidx]
                p_o = s_ords[partner]
                p_v = s_vals[partner]
                p_i = s_idxs[partner]
                cute.arch.barrier()

                # Comparator. Deterministic mode uses a lex
                # ``(ord, -idx)`` tie-break so equal-ord entries land in
                # ascending-idx order (matching aten). Non-deterministic
                # mode skips the tiebreak (cheaper compare-exchange).
                if const_expr(self.deterministic):
                    self_lt_partner = (my_o < p_o) or (my_o == p_o and my_i > p_i)
                    self_gt_partner = (my_o > p_o) or (my_o == p_o and my_i < p_i)
                else:
                    self_lt_partner = my_o < p_o
                    self_gt_partner = my_o > p_o
                block_dir = (tidx >> Int32(stage + 1)) & Int32(1)
                if tidx < K:
                    if tidx < partner:
                        if block_dir == Int32(0):
                            if self_lt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                        else:
                            if self_gt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                    else:
                        if block_dir == Int32(0):
                            if self_gt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                        else:
                            if self_lt_partner:
                                s_ords[tidx] = p_o
                                s_vals[tidx] = p_v
                                s_idxs[tidx] = p_i
                cute.arch.barrier()

        # Phase 4: results to gmem.
        if tidx < K:
            mValues[row, tidx] = s_vals[tidx]
            mIndices[row, tidx] = s_idxs[tidx]


def _make_fake_tensor(dtype, shape, divisibility=1):
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != len(shape) - 1 else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


@jit_cache
def _compile_topk_radix(N: int, K: int, deterministic: bool):
    batch_sym = cute.sym_int()
    div_n = math.gcd(4, N)
    div_k = math.gcd(4, K)
    x_fake = _make_fake_tensor(Float32, (batch_sym, N), div_n)
    v_fake = _make_fake_tensor(Float32, (batch_sym, K), div_k)
    i_fake = _make_fake_tensor(Int32, (batch_sym, K), div_k)
    return cute.compile(
        _RadixSelectTopK(N, K, deterministic=deterministic),
        x_fake,
        v_fake,
        i_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def topk_radix(
    x: torch.Tensor, k: int, *, deterministic: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused radix-select top-K for fp32 2D contiguous ``x``.

    Returns ``(values[M, K]: fp32, indices[M, K]: int64)`` with K sorted
    descending per row. Caller must ensure ``k`` is in the supported set
    (see ``cutedsl_impl.py``) and ``N % 4 == 0``.

    ``deterministic`` controls the gather + sort tie-breaking strategy:
    True (default) uses prefix-sum gather + lex comparator and matches
    aten bit-exactly; False uses smem atomic gather + ord-only sort and
    is faster but indices may differ on ties.
    """
    M, N = x.shape
    out_v = torch.empty(M, k, dtype=torch.float32, device=x.device)
    # Kernel writes int32 indices; widen before returning to match aten.
    out_i_i32 = torch.empty(M, k, dtype=torch.int32, device=x.device)
    _compile_topk_radix(N, k, deterministic)(x, out_v, out_i_i32)
    return out_v, out_i_i32.to(torch.int64)


# ---------------------------------------------------------------------------
# Register-resident top-K (small K, small N).
# ---------------------------------------------------------------------------


@cute.jit
def _make_key(val: Float32, idx: Int32) -> Int64:
    """Compose the descending-sort key: ``(ord_u32 << 32) | (~idx_u32)``.

    Sorting this Int64 descending gives ``(value desc, idx asc)`` -
    exactly aten's tie-break order. Real fp32 inputs (no NaN) keep keys
    away from INT64_MIN, leaving that bit pattern free as a sentinel.
    """
    ord32 = _f32_to_radix_ord(val)
    ord64 = Int64(ord32) & Int64(0xFFFFFFFF)
    inv_idx64 = Int64(~idx) & Int64(0xFFFFFFFF)
    return (ord64 << Int64(32)) | inv_idx64


@cute.jit
def _decode_key(key: Int64) -> tuple[Float32, Int32]:
    ord32 = Int32(key >> Int64(32))
    inv_idx = Int32(key & Int64(0xFFFFFFFF))
    val_bits = ord32 ^ ((ord32 >> Int32(31)) & Int32(0x7FFFFFFF))
    return Float32(llvm.bitcast(T.f32(), val_bits.ir_value())), ~inv_idx


@cute.jit
def _cas_desc(arr: cute.Tensor, i: cutlass.Constexpr, j: cutlass.Constexpr) -> None:
    a = arr[i]
    b = arr[j]
    if a < b:
        arr[i] = b
        arr[j] = a


@cute.jit
def _bitonic_sort_desc(arr: cute.Tensor, n: cutlass.Constexpr) -> None:
    if const_expr(n > 1):
        num_stages = int(math.log2(n))
        for s in cutlass.range_constexpr(num_stages):
            for sub_rev in cutlass.range_constexpr(s + 1):
                step = 1 << (s - sub_rev)
                for i in cutlass.range_constexpr(n):
                    j = i ^ step
                    if j > i:
                        block_dir = (i >> (s + 1)) & 1
                        if block_dir == 0:
                            _cas_desc(arr, i, j)
                        else:
                            a = arr[i]
                            b = arr[j]
                            if a > b:
                                arr[i] = b
                                arr[j] = a


@cute.jit
def _bitonic_merge_desc(arr: cute.Tensor, n: cutlass.Constexpr) -> None:
    """Merge a bitonic sequence of length n into descending sorted."""
    if const_expr(n > 1):
        num_levels = int(math.log2(n))
        for level in cutlass.range_constexpr(num_levels):
            length = n >> level
            step = length // 2
            for i in cutlass.range_constexpr(n // length):
                start_i = i * length
                for j in cutlass.range_constexpr(step):
                    _cas_desc(arr, start_i + j, start_i + j + step)


@cute.jit
def _topk_merge_desc(a: cute.Tensor, b: cute.Tensor, K: cutlass.Constexpr) -> None:
    """Top-K of two K-sorted-descending sequences, in place into ``a``.

    ``a[i] <- max(a[i], b[K-1-i])`` yields a bitonic sequence in ``a``;
    one bitonic merge of length K finishes it.
    """
    for i in cutlass.range_constexpr(K):
        x = a[i]
        y = b[K - 1 - i]
        if y > x:
            a[i] = y
    _bitonic_merge_desc(a, K)


class _RegisterTopK:
    """Register-resident top-K. Each warp owns one row; ``ROWS_PER_CTA``
    warps per CTA. K in {16, 32}, N a power of 2 in [K, 2048].
    """

    def __init__(self, N: int, K: int, rows_per_cta: int = 4):
        self.N = N
        self.K = K
        self.VEC = N // 32
        self.ROWS_PER_CTA = rows_per_cta
        self.NUM_THREADS = 32 * rows_per_cta

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M = mX.shape[0]
        ROWS = const_expr(self.ROWS_PER_CTA)
        num_blocks = (M + ROWS - 1) // ROWS
        self.kernel(mX, mValues, mIndices).launch(
            grid=[num_blocks, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        K = const_expr(self.K)
        VEC = const_expr(self.VEC)
        ROWS = const_expr(self.ROWS_PER_CTA)
        WARP_SIZE = const_expr(cute.arch.WARP_SIZE)

        warp_idx = tidx // Int32(WARP_SIZE)
        lane_idx = tidx % Int32(WARP_SIZE)
        row = Int32(bidx) * Int32(ROWS) + warp_idx

        M = mX.shape[0]
        # CuTeDSL can't early-return on a dynamic predicate. Clamp the row
        # for loads to keep them in-bounds, and gate the writes only.
        row_safe = row if row < M else Int32(0)
        in_bounds = row < M

        # Per-thread Int64 keys covering this row's VEC-sized stripe.
        keys = cute.make_rmem_tensor(VEC, Int64)
        for i in cutlass.range_constexpr(VEC):
            col = lane_idx * Int32(VEC) + Int32(i)
            keys[i] = _make_key(mX[row_safe, col], col)

        _bitonic_sort_desc(keys, VEC)

        # Build per-thread top-K. If VEC < K, pad with INT64_MIN so the
        # sentinel sorts strictly below any real key; if VEC >= K take the
        # leading K (already sorted).
        topk = cute.make_rmem_tensor(K, Int64)
        if const_expr(VEC >= K):
            for i in cutlass.range_constexpr(K):
                topk[i] = keys[i]
        else:
            SENTINEL = const_expr(-(1 << 63))
            for i in cutlass.range_constexpr(VEC):
                topk[i] = keys[i]
            for i in cutlass.range_constexpr(K - VEC):
                topk[VEC + i] = Int64(SENTINEL)
            _bitonic_sort_desc(topk, K)

        # Warp-cooperative top-K merge via butterfly shuffles.
        log2_warp = const_expr(int(math.log2(WARP_SIZE)))
        for s in cutlass.range_constexpr(log2_warp):
            other = cute.make_rmem_tensor(K, Int64)
            for i in cutlass.range_constexpr(K):
                other[i] = cute.arch.shuffle_sync_bfly(topk[i], offset=Int32(1 << s))
            _topk_merge_desc(topk, other, K)

        # Lane 0 of each warp now holds the row's top-K descending.
        if lane_idx == Int32(0) and in_bounds:
            for i in cutlass.range_constexpr(K):
                v, idx = _decode_key(topk[i])
                mValues[row, i] = v
                mIndices[row, i] = idx


@jit_cache
def _compile_topk_register(N: int, K: int):
    batch_sym = cute.sym_int()
    div_n = math.gcd(4, N)
    div_k = math.gcd(4, K)
    x_fake = _make_fake_tensor(Float32, (batch_sym, N), div_n)
    v_fake = _make_fake_tensor(Float32, (batch_sym, K), div_k)
    i_fake = _make_fake_tensor(Int32, (batch_sym, K), div_k)
    return cute.compile(
        _RegisterTopK(N, K),
        x_fake,
        v_fake,
        i_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def topk_register(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Register-resident top-K for fp32 2D contiguous ``x``.

    Returns ``(values[M, K]: fp32, indices[M, K]: int64)``, descending,
    bit-exact to aten on both values and indices. Caller must ensure
    ``k`` in {16, 32} and ``N`` is a power of 2 in [k, 2048].
    """
    M, N = x.shape
    out_v = torch.empty(M, k, dtype=torch.float32, device=x.device)
    out_i_i32 = torch.empty(M, k, dtype=torch.int32, device=x.device)
    _compile_topk_register(N, k)(x, out_v, out_i_i32)
    return out_v, out_i_i32.to(torch.int64)
