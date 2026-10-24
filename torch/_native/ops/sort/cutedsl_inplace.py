"""In-place leaves: mirror ``sortKeyValueInplace`` for rows of at most 4096.

Stock ``Sort.cpp`` copies ``self`` into ``values``, fills ``indices`` with iota
and sorts both in place. These kernels read ``x`` and write ``values`` and the
out= index dtype directly, with the leaf and launch shape stock picks by row
length (``sort_size``):

- 32, unstable only: ``bitonicSortKVInPlace``, 16 threads x 2 items per row and
  16 rows per CTA. Its compare-exchange network runs in registers, so ties
  resolve exactly as in stock.
- 128: ``warpMergeSortKVInPlace``, one warp x 4 items per row and 16 rows per
  CTA. Its result is the unique stable order, computed by the same network on
  keys made unique by their index.
- 1024 / 2048 / 4096: ``radixSortKVInPlace``, ``sort_size / 32`` threads x 32
  items, one row per CTA; a stable LSD radix sort over 8-bit digits with the
  row held in registers and exchanged through shared memory each pass.

Stock uses these shapes for every key width, 8-byte keys included.

The comparison leaves use ``LTOp`` / ``GTOp``: every NaN ties with every other
NaN above +inf and ``-0.0 == +0.0``. The radix leaf orders keys like
``radix_ordinal``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch
from torch._native.cutedsl.launch import compile_kernel, read_only
from torch._native.cutedsl.scan import block_exclusive_prefix
from torch._native.instrumentation import instrumented_cutedsl_cache

from .cutedsl_common import (
    fake_rows,
    INDEX_DTYPES,
    index_name,
    KEY_KINDS,
    RADIX_BINS,
    RADIX_BITS,
    radix_ordinal,
    RAW_CUTE,
    raw_view,
)


# Bits of +inf; larger magnitudes are NaN.
_INF_BITS = {
    "float16": 0x7C00,
    "bfloat16": 0x7F80,
    "float32": 0x7F800000,
    "float64": 0x7FF0000000000000,
}
# max_block_y of the bitonic and warp merge leaves.
_ROWS_PER_CTA = 16


def sort_size(nsort: int, stable: bool) -> int:
    """Stock leaf for a row of ``nsort <= 4096``, as its padded sort size."""
    if not stable and nsort <= 32:
        return 32
    if nsort <= 128:
        return 128
    return max(1024, 1 << (nsort - 1).bit_length())


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
        self.sort_size = sort_size
        if sort_size == 32:
            self.block = (16, _ROWS_PER_CTA)
        elif sort_size == 128:
            self.block = (32, _ROWS_PER_CTA)
        else:
            self.block = (sort_size // 32, 1)

    @cute.jit
    def __call__(
        self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor, stream
    ):
        rows_per_cta = const_expr(self.block[1])
        self.kernel(mX, mValues, mIndices).launch(
            grid=[(mX.shape[0] + rows_per_cta - 1) // rows_per_cta, 1, 1],
            block=[self.block[0], self.block[1], 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        if const_expr(self.sort_size <= 128):
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

        The unstable leaf compares ordinals only, so ties swap exactly as in
        stock. The stable leaf also compares indices, so keys are unique and
        the network yields the stable order.
        """
        if const_expr(self.key_bits <= 32):
            shift = const_expr(IDX_BITS if self.sort_size == 32 else 0)
            return (ka >> shift) < (kb >> shift)
        if const_expr(self.sort_size == 32):
            return ka < kb or (wb >> IDX_BITS) != Int32(0)
        return ka < kb or (ka == kb and wa < wb)

    @cute.jit
    def radix(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        """cub BlockRadixSort shape; warp w, item k, lane l owns element
        32 * (ITEMS * w + k) + l, so ranking in (w, k, l) order is stable."""
        ITEMS = const_expr(32)
        NUM_THREADS = const_expr(self.block[0])
        NUM_WARPS = const_expr(NUM_THREADS // 32)
        PASSES = const_expr(self.key_bits // RADIX_BITS)
        DIGITS_PER_THREAD = const_expr(RADIX_BINS // NUM_THREADS)
        DIGIT_MASK = const_expr(RADIX_BINS - 1)
        desc = const_expr(self.descending)
        # Keys up to 16 bits travel with their index in one 32-bit word.
        packed = const_expr(self.key_bits <= 16)
        key_mask = Int32((1 << self.key_bits) - 1) if const_expr(packed) else Int32(-1)
        word_t = Int64 if const_expr(self.key_bits == 64) else Int32
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        n = mX.shape[1]
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)

        smem = cutlass.utils.SmemAllocator()
        s_count = smem.allocate_tensor(
            Int32, cute.make_layout((NUM_WARPS * RADIX_BINS,)), byte_alignment=16
        )
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((NUM_WARPS + 1,)), byte_alignment=16
        )
        s_buf = smem.allocate_tensor(
            word_t, cute.make_layout((self.sort_size,)), byte_alignment=16
        )

        warp_base = warp * Int32(32 * ITEMS)
        bits = cute.make_rmem_tensor(ITEMS, word_t)
        index = cute.make_rmem_tensor(ITEMS, Int32)
        digit = cute.make_rmem_tensor(ITEMS, Int32)
        rank = cute.make_rmem_tensor(ITEMS, Int32)
        for k in cutlass.range_constexpr(ITEMS):
            j = warp_base + Int32(k * 32) + lane
            bits[k] = word_t(0)
            index[k] = j
            if j < n:
                bits[k] = word_t(mX[row, j])

        for p in cutlass.range_constexpr(PASSES):
            for i in cutlass.range_constexpr(NUM_WARPS * RADIX_BINS // NUM_THREADS):
                s_count[tidx + i * NUM_THREADS] = Int32(0)
            cute.arch.barrier()

            for k in cutlass.range_constexpr(ITEMS):
                if warp_base + Int32(k * 32) < n:
                    j = warp_base + Int32(k * 32) + lane
                    digit[k] = Int32(RADIX_BINS)  # inactive lanes match only each other
                    if j < n:
                        ord_ = radix_ordinal(bits[k], self.kind, self.key_bits, desc)
                        digit[k] = Int32(
                            (ord_ >> word_t(p * RADIX_BITS)) & word_t(DIGIT_MASK)
                        )
                    peers = Int32(cute.arch.match_sync(0xFFFFFFFF, digit[k]))
                    earlier = peers & Int32(cute.arch.lanemask_lt())
                    rank[k] = cute.arch.popc(earlier)
                    slot = warp * RADIX_BINS + digit[k]
                    if j < n:
                        rank[k] = rank[k] + s_count[slot]
                    cute.arch.sync_warp()
                    if j < n and earlier == Int32(0):
                        s_count[slot] = rank[k] + cute.arch.popc(peers)
                    cute.arch.sync_warp()
            cute.arch.barrier()

            # Thread t owns digits [t * D, (t + 1) * D): per digit, exclusive
            # offsets over warps, then an exclusive scan of the digit totals.
            first = tidx * Int32(DIGITS_PER_THREAD)
            totals = cute.make_rmem_tensor(DIGITS_PER_THREAD, Int32)
            chunk = Int32(0)
            for d in cutlass.range_constexpr(DIGITS_PER_THREAD):
                running = Int32(0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    c = s_count[w * RADIX_BINS + first + d]
                    s_count[w * RADIX_BINS + first + d] = running
                    running = running + c
                totals[d] = running
                chunk = chunk + running
            excl, _ = block_exclusive_prefix(chunk, s_scan, lane, warp, NUM_WARPS)
            for d in cutlass.range_constexpr(DIGITS_PER_THREAD):
                for w in cutlass.range_constexpr(NUM_WARPS):
                    s_count[w * RADIX_BINS + first + d] = (
                        s_count[w * RADIX_BINS + first + d] + excl
                    )
                excl = excl + totals[d]
            cute.arch.barrier()

            for k in cutlass.range_constexpr(ITEMS):
                if warp_base + Int32(k * 32) + lane < n:
                    rank[k] = rank[k] + s_count[warp * RADIX_BINS + digit[k]]
            for r in cutlass.range_constexpr(1 if packed else 2):
                for k in cutlass.range_constexpr(ITEMS):
                    if warp_base + Int32(k * 32) + lane < n:
                        if const_expr(packed):
                            s_buf[rank[k]] = (bits[k] << Int32(16)) | index[k]
                        else:
                            s_buf[rank[k]] = (
                                bits[k] if const_expr(r == 0) else word_t(index[k])
                            )
                cute.arch.barrier()
                for k in cutlass.range_constexpr(ITEMS):
                    j = warp_base + Int32(k * 32) + lane
                    if j < n:
                        word = s_buf[j]
                        if const_expr(packed):
                            bits[k] = (word >> Int32(16)) & key_mask
                            index[k] = word & Int32(0xFFFF)
                        elif const_expr(r == 0):
                            bits[k] = word
                        else:
                            index[k] = Int32(word)
                cute.arch.barrier()

        for k in cutlass.range_constexpr(ITEMS):
            j = warp_base + Int32(k * 32) + lane
            if j < n:
                mValues[row, j] = self.raw_dtype(bits[k])
                mIndices[row, j] = self.index_dtype(index[k])


def spec_name(spec: dict) -> str:
    return (
        f"sort_inplace{spec['sort_size']}_{spec['key']}_{spec['index']}"
        f"_{'desc' if spec['descending'] else 'asc'}_sm{spec['arch']}"
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

    The caller checks eligibility: key and index dtypes supported, row length
    at most 4096 and addressable by the index dtype, contiguous outputs.
    """
    if x.numel() == 0:
        return
    spec = make_spec(
        arch, x.dtype, indices.dtype, descending, sort_size(x.shape[1], stable)
    )
    _compile(tuple(sorted(spec.items())))(
        read_only(raw_view(x)), raw_view(values), indices
    )
