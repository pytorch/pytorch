"""In-place leaves: mirror ``sortKeyValueInplace`` for the rows policy.route
sends here (the register kernel takes the other rows of at most 2048).

Stock ``Sort.cpp`` copies ``self`` into ``values``, fills ``indices`` with iota
and sorts both in place. These kernels read ``x`` and write ``values`` and the
out= index dtype directly, with the leaf and launch shape stock picks by row
length (``sort_size``):

- 32, unstable rows of at most 32: ``bitonicSortKVInPlace``, 16 threads x 2
  items per row and 16 rows per CTA. Its compare-exchange network runs in
  registers, so ties resolve exactly as in stock.
- 4096, rows of 2049 .. 4096: ``radixSortKVInPlace``, 128 threads x 32 items,
  one row per CTA: cub ``BlockRadixSort`` with its defaults (4-bit digits,
  packed-counter ranking, padded exchanges, transpose load and store).

Stock uses these shapes for every key width, 8-byte keys included.

The comparison leaves use ``LTOp`` / ``GTOp``: every NaN ties with every other
NaN above +inf and ``-0.0 == +0.0``. The radix leaf orders keys like
``radix_ordinal``.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, Uint16, Uint8

import torch
from torch._native.cutedsl.launch import compile_kernel, read_only
from torch._native.cutedsl.scan import block_exclusive_prefix
from torch._native.instrumentation import instrumented_cutedsl_cache

from .cutedsl_common import (
    fake_rows,
    INDEX_DTYPES,
    index_name,
    KEY_KINDS,
    radix_ordinal,
    radix_ordinal_inverse,
    RAW_CUTE,
    raw_view,
)
from .policy import BITONIC_MAX_NSORT, INPLACE_MAX_NSORT, inplace_sort_size


# Bits of +inf; larger magnitudes are NaN.
_INF_BITS = {
    "float16": 0x7C00,
    "bfloat16": 0x7F80,
    "float32": 0x7F800000,
    "float64": 0x7FF0000000000000,
}
# max_block_y of the bitonic leaf.
_ROWS_PER_CTA = 16
# Registers per thread that ptxas gives stock's 128-thread radixSortKVInPlace
# for keys and narrow values of at most 4 bytes (cuobjdump -res-usage); the
# radix leaf asks for the residency that implies. 8-byte keys get no budget, as
# stock's run near 255 there.
_STOCK_RADIX_REGISTERS = 168
_REGISTERS_PER_SM = 65536


@cute.jit
def _cmp_ordinal(
    bits: Int32 | Int64,
    key: cutlass.Constexpr,
    kind: cutlass.Constexpr,
    key_bits: cutlass.Constexpr,
    descending: cutlass.Constexpr,
) -> Int32 | Int64:
    """Ordinal whose order is ``LTOp`` (``GTOp`` when descending); Int64 for
    64-bit keys, like ``radix_ordinal``."""
    if const_expr(kind == "float"):
        word = Int64 if const_expr(key_bits == 64) else Int32
        magnitude = word((1 << (key_bits - 1)) - 1)
        if bits & magnitude > word(_INF_BITS[key]):
            bits = magnitude
    return radix_ordinal(bits, kind, key_bits, descending)


class InplaceSort:
    def __init__(
        self,
        key: str,
        index_dtype: type[cutlass.Numeric],
        descending: bool,
        sort_size: int,
    ):
        self.key = key
        self.kind, self.key_bits = KEY_KINDS[getattr(torch, key)]
        self.raw_dtype = RAW_CUTE[self.key_bits]
        self.index_dtype = index_dtype
        self.descending = descending
        if sort_size not in (BITONIC_MAX_NSORT, INPLACE_MAX_NSORT):
            raise ValueError(f"no in-place leaf of sort size {sort_size}")
        self.sort_size = sort_size
        bitonic = sort_size == BITONIC_MAX_NSORT
        self.block = (16, _ROWS_PER_CTA) if bitonic else (sort_size // 32, 1)
        threads = self.block[0] * self.block[1]
        self.min_blocks = (
            0
            if bitonic or self.key_bits > 32
            else _REGISTERS_PER_SM // (_STOCK_RADIX_REGISTERS * threads)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows_per_cta = const_expr(self.block[1])
        self.kernel(mX, mValues, mIndices).launch(
            # Rows are at least 1 (empty sorts return before launch) and up to
            # policy.MAX_ROWS; rounding up as (M + r - 1) // r would wrap Int32.
            grid=[(mX.shape[0] - 1) // rows_per_cta + 1, 1, 1],
            block=[self.block[0], self.block[1], 1],
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            # Without a preference the driver sizes the carveout for min_blocks
            # only, which leaves the radix leaf short of shared memory.
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        if const_expr(self.sort_size == BITONIC_MAX_NSORT):
            self.bitonic(mX, mValues, mIndices)
        else:
            self.radix(mX, mValues, mIndices)

    @cute.jit
    def bitonic(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        """bitonicSort of SortUtils.cuh in registers; lane l of a row owns
        elements ITEMS * l .. ITEMS * (l + 1) - 1, see ``less`` for the keys."""
        LANES = const_expr(self.block[0])
        ITEMS = const_expr(self.sort_size // LANES)
        IDX_BITS = const_expr((self.sort_size - 1).bit_length())
        wide = const_expr(self.key_bits == 64)
        key_t = Int64 if const_expr(self.key_bits >= 32) else Int32
        lane, ty, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * Int32(_ROWS_PER_CTA) + ty
        n = mX.shape[1]
        row_valid = row < mX.shape[0]

        # Keys up to 32 bits pack ``ordinal << IDX_BITS | index``, an invalid
        # key's ordinal field above every valid one. 64-bit keys keep the
        # sign-flipped ordinal (invalid: max) and ``index | invalid << IDX_BITS``.
        keys = cute.make_rmem_tensor(ITEMS, key_t)
        words = cute.make_rmem_tensor(ITEMS, Int32)
        for i in cutlass.range_constexpr(ITEMS):
            j = lane * Int32(ITEMS) + Int32(i)
            if const_expr(wide):
                keys[i] = Int64((1 << 63) - 1)
                words[i] = j | Int32(1 << IDX_BITS)
            else:
                keys[i] = key_t(1 << (self.key_bits + IDX_BITS)) | key_t(j)
            if row_valid and j < n:
                ord_ = _cmp_ordinal(
                    (Int64 if const_expr(wide) else Int32)(mX[row, j]),
                    self.key,
                    self.kind,
                    self.key_bits,
                    self.descending,
                )
                if const_expr(wide):
                    keys[i] = ord_ ^ Int64(-(1 << 63))
                    words[i] = j
                else:
                    keys[i] = ((key_t(ord_) & key_t(0xFFFFFFFF)) << IDX_BITS) | key_t(j)

        # Pair (e, e + stride) swaps iff less(e, e + stride) == (e & size != 0).
        for log_size in cutlass.range_constexpr(1, IDX_BITS + 1):
            size = 1 << log_size
            for log_stride in cutlass.range_constexpr(log_size - 1, -1, -1):
                stride = 1 << log_stride
                if const_expr(stride < ITEMS):
                    for i in cutlass.range_constexpr(ITEMS):
                        if const_expr((i & stride) == 0):
                            ka, wa = keys[i], words[i]
                            kb, wb = keys[i + stride], words[i + stride]
                            e = lane * Int32(ITEMS) + Int32(i)
                            flag = (e & Int32(size)) != Int32(0)
                            if self.less(ka, wa, kb, wb, IDX_BITS) == flag:
                                keys[i], keys[i + stride] = kb, ka
                                if const_expr(wide):
                                    words[i], words[i + stride] = wb, wa
                else:
                    lane_stride = stride // ITEMS
                    flag = (lane & Int32(size // ITEMS)) != Int32(0)
                    lower = (lane & Int32(lane_stride)) == Int32(0)
                    for i in cutlass.range_constexpr(ITEMS):
                        k, w = keys[i], words[i]
                        ko = cute.arch.shuffle_sync_bfly(k, offset=lane_stride)
                        wo = w
                        if const_expr(wide):
                            wo = cute.arch.shuffle_sync_bfly(w, offset=lane_stride)
                        less = self.less(k, w, ko, wo, IDX_BITS)
                        if not lower:
                            less = self.less(ko, wo, k, w, IDX_BITS)
                        if less == flag:
                            keys[i] = ko
                            words[i] = wo

        for i in cutlass.range_constexpr(ITEMS):
            j = lane * Int32(ITEMS) + Int32(i)
            if row_valid and j < n:
                src = words[i] if const_expr(wide) else keys[i]
                src = Int32(src & key_t((1 << IDX_BITS) - 1))
                mValues[row, j] = mX[row, src]
                mIndices[row, j] = self.index_dtype(src)

    @cute.jit
    def less(self, ka, wa: Int32, kb, wb: Int32, IDX_BITS: cutlass.Constexpr):
        """bitonicSwap's ``comp(kA, kB) && validA || !validB`` on encoded keys.

        Only ordinals are compared, so ties swap exactly as in stock.
        """
        if const_expr(self.key_bits <= 32):
            return (ka >> IDX_BITS) < (kb >> IDX_BITS)
        return ka < kb or (wb >> IDX_BITS) != Int32(0)

    @cute.jit
    def _digit(self, ord_, neg_zero, pos_zero, shift, bits: cutlass.Constexpr) -> Int32:
        """BFEDigitExtractor over a bit-ordered key (ProcessFloatMinusZero first)."""
        if const_expr(self.kind == "float"):
            if ord_ == neg_zero:
                ord_ = pos_zero
        word_t = Int64 if const_expr(self.key_bits == 64) else Int32
        return Int32((ord_ >> shift) & word_t((1 << bits) - 1))

    @cute.jit
    def radix(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        """radixSortKVInPlace: cub BlockLoad / BlockRadixSort / BlockStore with
        their defaults (4-bit digits, BlockRadixRank's packed 16-bit counters and
        raking scan, padded BlockExchange). Thread t holds the blocked items
        t * ITEMS .. t * ITEMS + ITEMS - 1. Like stock, the row is padded with
        keys that sort last (all-ones bit-ordered); the index is the position, so
        no iota."""
        ITEMS = const_expr(32)
        THREADS = const_expr(self.block[0])
        WARPS = const_expr(THREADS // 32)
        BITS = const_expr(4)
        PASSES = const_expr(self.key_bits // BITS)
        COUNTER_LANES = const_expr(8)  # (1 << BITS) digits / 2 counters per word
        RAKING_SEGMENT = const_expr(COUNTER_LANES + 1)  # + 1 lane of bank padding
        desc = const_expr(self.descending)
        word_t = Int64 if const_expr(self.key_bits == 64) else Int32
        key_mask = (
            word_t(-1)
            if const_expr(self.key_bits >= 32)
            else word_t((1 << self.key_bits) - 1)
        )
        index_bits = const_expr(self.index_dtype.width)
        index_mask = (
            Int32(-1) if const_expr(index_bits >= 32) else Int32((1 << index_bits) - 1)
        )
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        n = mX.shape[1]
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)

        # cub's TempStorage is a union of the ranking counters and the key and
        # value exchanges (and stock unions BlockLoad / BlockStore over it), so
        # shared memory per CTA, and with it residency, matches stock.
        padded = const_expr(
            self.sort_size + self.sort_size // 32
        )  # BlockExchange padding
        key_bytes = const_expr(self.key_bits // 8)
        index_bytes = const_expr(index_bits // 8)
        ranking_bytes = const_expr(4 * (RAKING_SEGMENT * THREADS + WARPS + 1))
        union_bytes = const_expr(
            max(ranking_bytes, padded * max(key_bytes, index_bytes))
        )
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate_tensor(
            Uint8, cute.make_layout((union_bytes,)), byte_alignment=16
        )

        def view(dtype, size, offset=0):
            ptr = cute.recast_ptr(storage.iterator + offset, dtype=dtype)
            return cute.make_tensor(ptr, cute.make_layout((size,)))

        s_counters = view(Int32, RAKING_SEGMENT * THREADS)  # packed, for the scan
        s_digit_counters = view(
            Uint16, 2 * RAKING_SEGMENT * THREADS
        )  # [lane][thread][half]
        s_scan = view(Int32, WARPS + 1, 4 * RAKING_SEGMENT * THREADS)
        s_keys = view(self.raw_dtype, padded)
        s_indices = view(self.index_dtype, padded)

        bits = cute.make_rmem_tensor(ITEMS, word_t)
        index = cute.make_rmem_tensor(ITEMS, Int32)
        rank = cute.make_rmem_tensor(ITEMS, Int32)
        segment = cute.make_rmem_tensor(RAKING_SEGMENT, Int32)
        blocked = tidx * Int32(ITEMS)
        # Padded smem slots (one per 32 items): blocked item k of thread t sits at
        # 33 t + k, striped item k at k (T + T / 32) + t + t / 32.
        blocked_pad = tidx * Int32(ITEMS + 1)
        striped_pad = tidx + (tidx >> Int32(5))
        STRIPED_STEP = const_expr(THREADS + THREADS // 32)

        # BLOCK_LOAD_TRANSPOSE: coalesced striped loads, then blocked via smem.
        for k in cutlass.range_constexpr(ITEMS):
            j = Int32(k * THREADS) + tidx
            if j < n:
                s_keys[striped_pad + Int32(k * STRIPED_STEP)] = mX[row, j]
        cute.arch.barrier()
        # Keys stay bit-ordered (to_bit_ordered) between the load and the store;
        # -0.0 maps to +0.0 only when a digit is extracted, as cub does.
        for k in cutlass.range_constexpr(ITEMS):
            raw = word_t(s_keys[blocked_pad + Int32(k)]) & key_mask
            bits[k] = radix_ordinal(
                raw, self.kind, self.key_bits, desc, keep_negative_zero=True
            )
            j = blocked + Int32(k)
            if j >= n:
                bits[k] = key_mask
            index[k] = j
        cute.arch.barrier()
        neg_zero = radix_ordinal(
            word_t(1 << (self.key_bits - 1)),
            self.kind,
            self.key_bits,
            desc,
            keep_negative_zero=True,
        )
        pos_zero = radix_ordinal(word_t(0), self.kind, self.key_bits, desc)

        for p in cutlass.range_constexpr(PASSES):
            shift_p = word_t(p * BITS)
            # BlockRadixRank::RankKeys. Thread t owns column t of every counter
            # lane, so counting needs no atomics.
            for c in cutlass.range_constexpr(RAKING_SEGMENT):
                s_counters[c * THREADS + tidx] = Int32(0)
            for k in cutlass.range_constexpr(ITEMS):
                digit = self._digit(bits[k], neg_zero, pos_zero, shift_p, BITS)
                counter = (
                    ((digit & Int32(COUNTER_LANES - 1)) * Int32(THREADS) + tidx)
                    << Int32(1)
                ) | (digit >> Int32(3))
                prefix = Int32(s_digit_counters[counter])
                s_digit_counters[counter] = Uint16(prefix + Int32(1))
                # cub keeps digit_counters[ITEM] (a pointer) beside the prefix; the
                # prefix is below 2^16, so the counter offset rides in the high half.
                rank[k] = prefix | (counter << Int32(16))
            cute.arch.barrier()

            # ScanCounters: raking upsweep over this thread's segment, a block
            # exclusive scan of the packed partials (the low half's total carried
            # into the high half), then the downsweep back into smem.
            partial = Int32(0)
            for c in cutlass.range_constexpr(RAKING_SEGMENT):
                segment[c] = s_counters[tidx * Int32(RAKING_SEGMENT) + Int32(c)]
                partial = partial + segment[c]
            excl, total = block_exclusive_prefix(partial, s_scan, lane, warp, WARPS)
            running = excl + (total << Int32(16))
            for c in cutlass.range_constexpr(RAKING_SEGMENT):
                s_counters[tidx * Int32(RAKING_SEGMENT) + Int32(c)] = running
                running = running + segment[c]
            cute.arch.barrier()

            for k in cutlass.range_constexpr(ITEMS):
                c = rank[k]
                rank[k] = (c & Int32(0xFFFF)) + Int32(s_digit_counters[c >> Int32(16)])
            cute.arch.barrier()

            # BlockExchange::ScatterToBlocked, keys then values.
            for k in cutlass.range_constexpr(ITEMS):
                r = rank[k]
                s_keys[r + (r >> Int32(5))] = self.raw_dtype(bits[k])
            cute.arch.barrier()
            for k in cutlass.range_constexpr(ITEMS):
                bits[k] = word_t(s_keys[blocked_pad + Int32(k)]) & key_mask
            cute.arch.barrier()
            for k in cutlass.range_constexpr(ITEMS):
                r = rank[k]
                s_indices[r + (r >> Int32(5))] = self.index_dtype(index[k])
            cute.arch.barrier()
            for k in cutlass.range_constexpr(ITEMS):
                index[k] = Int32(s_indices[blocked_pad + Int32(k)]) & index_mask
            cute.arch.barrier()

        # BLOCK_STORE_TRANSPOSE: blocked to striped via smem, coalesced stores;
        # keys, then values.
        for k in cutlass.range_constexpr(ITEMS):
            raw = radix_ordinal_inverse(bits[k], self.kind, self.key_bits, desc)
            s_keys[blocked_pad + Int32(k)] = self.raw_dtype(raw)
        cute.arch.barrier()
        for k in cutlass.range_constexpr(ITEMS):
            j = Int32(k * THREADS) + tidx
            if j < n:
                mValues[row, j] = s_keys[striped_pad + Int32(k * STRIPED_STEP)]
        cute.arch.barrier()
        for k in cutlass.range_constexpr(ITEMS):
            s_indices[blocked_pad + Int32(k)] = self.index_dtype(index[k])
        cute.arch.barrier()
        for k in cutlass.range_constexpr(ITEMS):
            j = Int32(k * THREADS) + tidx
            if j < n:
                mIndices[row, j] = s_indices[striped_pad + Int32(k * STRIPED_STEP)]


def spec_name(spec: dict) -> str:
    return (
        f"sort_inplace{spec['sort_size']}_{spec['key']}_{spec['index']}"
        f"_{'desc' if spec['descending'] else 'asc'}"
        + (f"_sm{spec['arch']}" if "arch" in spec else "")
    )


def make_spec(
    arch: int,
    key_dtype: torch.dtype,
    index_dtype: torch.dtype,
    descending: bool,
    sort_size: int,
) -> dict:
    return {
        "kernel": "inplace",
        "key": index_name(key_dtype),
        "index": index_name(index_dtype),
        "descending": descending,
        "sort_size": sort_size,
        "arch": arch,
    }


def compile_args(spec: dict) -> tuple[InplaceSort, list]:
    """Kernel object and fake operands for one spec; shared by JIT and AOT."""
    raw = RAW_CUTE[KEY_KINDS[getattr(torch, spec["key"])][1]]
    index = INDEX_DTYPES[getattr(torch, spec["index"])]
    kernel = InplaceSort(spec["key"], index, spec["descending"], spec["sort_size"])
    fake = [
        fake_rows(raw),
        fake_rows(raw),
        fake_rows(index),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return kernel, fake


@instrumented_cutedsl_cache("aten::sort", key_fn=lambda spec: spec_name(dict(spec)))
def _compile(spec: tuple):
    kernel, fake = compile_args(dict(spec))
    return compile_kernel(kernel, *fake)


def inplace_sort(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    descending: bool,
    stable: bool,
    arch: int,
) -> None:
    """Sort each row of 2D row-major ``x`` into ``values`` and ``indices``.

    The caller checks eligibility: key and index dtypes supported, rows that
    policy.route sends here (unstable of at most 32, or 2049 .. 4096) and the
    index dtype addresses, contiguous outputs.
    """
    if x.numel() == 0:
        return
    spec = make_spec(
        arch, x.dtype, indices.dtype, descending, inplace_sort_size(x.shape[1], stable)
    )
    _compile(tuple(sorted(spec.items())))(
        read_only(raw_view(x)), raw_view(values), indices
    )
