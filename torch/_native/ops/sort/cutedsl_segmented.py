"""Segmented radix leaf: mirrors ``segmented_sort_pairs`` (cub DeviceSegmentedRadixSort).

One CTA sorts one row per digit pass (DeviceSegmentedRadixSortKernel in
device/dispatch/kernels/radix_sort.cuh): AgentRadixSortUpsweep counts the
row's digits in packed 8-bit thread counters, a block scan turns them into
offsets, and AgentRadixSortDownsweep ranks each tile with RADIX_RANK_MEMOIZE
(BlockRadixRank: packed 16-bit thread counters and a raking scan) after a
BLOCK_LOAD_TRANSPOSE load, then scatters keys and values through shared memory.
Passes alternate between cub's SegmentedPolicy and AltSegmentedPolicy exactly
as DispatchSegmentedRadixSort::InvokePasses schedules them.

Deviations from the stock code: the first pass generates the index instead of
reading reverse_indices; values and narrow indices are written straight to the
outputs; the alternate buffers cover a chunk of rows, not the tensor.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
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
    radix_ordinal,
    RAW_CUTE,
    RAW_TORCH,
    raw_view,
)
from .policy import segmented_policies


# Chunk of rows sharing the alternate buffers: at least 128 rows so a chunk
# keeps the stock leaf, otherwise at most 32M elements.
CHUNK_ITEMS = 1 << 25
_MIN_CHUNK_ROWS = 128


def pass_schedule(
    key_bits: int, radix_bits: int, alt_bits: int
) -> list[tuple[int, int, bool]]:
    """(current bit, pass bits, alternate policy) of InvokePasses."""
    num_passes = max(-(-key_bits // radix_bits), 1)
    alt_end = min(key_bits, (num_passes * radix_bits - key_bits) * alt_bits)
    schedule = []
    bit = 0
    while bit < key_bits:
        alt = bit < alt_end
        bits = min(alt_bits if alt else radix_bits, key_bits - bit)
        schedule.append((bit, bits, alt))
        bit += bits
    return schedule


class SegmentedPass:
    """One digit pass of DeviceSegmentedRadixSortKernel under one policy."""

    def __init__(
        self,
        kind: str,
        key_bits: int,
        index_dtype: type[cutlass.Numeric],
        descending: bool,
        policy: tuple[int, int, int],
    ):
        self.kind = kind
        self.key_bits = key_bits
        self.word = Int64 if key_bits == 64 else Int32
        self.raw_dtype = RAW_CUTE[key_bits]
        self.index_dtype = index_dtype
        self.descending = descending
        self.threads, self.items, self.radix_bits = policy
        self.tile = self.threads * self.items
        self.digits = 1 << self.radix_bits
        self.warps = self.threads // 32
        # AgentRadixSortUpsweep: four 8-bit counters per word, flushed every
        # UNROLL_COUNT tiles.
        self.up_lanes = 1 << max(0, self.radix_bits - 2)
        self.up_lanes_per_warp = max(1, -(-self.up_lanes // self.warps))
        self.unroll = min(64, 255 // self.items)
        # BlockRadixRank: two 16-bit counters per word, one padding lane.
        self.rank_lanes = 1 << max(0, self.radix_bits - 1)
        self.raking = self.rank_lanes + 1
        # Raw bits of the largest ordinal: cub's out-of-bounds key.
        mask = (1 << key_bits) - 1
        sign = 1 << (key_bits - 1)
        top = 0 if descending else mask
        if kind == "signed":
            pad = top ^ sign
        elif kind == "float":
            pad = top ^ sign if top & sign else ~top & mask
        else:
            pad = top
        self.pad = (
            pad - (1 << key_bits) if key_bits >= 32 and pad >> (key_bits - 1) else pad
        )
        exchange = self.tile * max(key_bits, index_dtype.width) // 32 + 1
        self.union_words = max(
            exchange,
            self.raking * self.threads,
            self.up_lanes * self.threads,
            32 * self.digits,
        )

    @cute.jit
    def digit(self, key, current_bit: cutlass.Constexpr, pass_bits: cutlass.Constexpr):
        ord_ = radix_ordinal(key, self.kind, self.key_bits, self.descending)
        return Int32((ord_ >> current_bit) & ((1 << pass_bits) - 1))

    @cute.kernel
    def kernel(
        self,
        mKeysIn: cute.Tensor,
        mIdxIn: cute.Tensor,
        mKeysOut: cute.Tensor,
        mIdxOut: cute.Tensor,
        row_base: Int32,
        current_bit: cutlass.Constexpr,
        pass_bits: cutlass.Constexpr,
        first: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        n = mKeysIn.shape[1]
        THREADS = const_expr(self.threads)
        ITEMS = const_expr(self.items)
        TILE = const_expr(self.tile)
        DIGITS = const_expr(self.digits)
        WARPS = const_expr(self.warps)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)

        smem = cutlass.utils.SmemAllocator()
        s_union = smem.allocate_tensor(
            Int32, cute.make_layout((self.union_words,)), byte_alignment=16
        )
        s_bytes = cute.make_tensor(
            cute.recast_ptr(s_union.iterator, dtype=cutlass.Uint8),
            cute.make_layout((4 * self.union_words,)),
        )
        s_halves = cute.make_tensor(
            cute.recast_ptr(s_union.iterator, dtype=cutlass.Uint16),
            cute.make_layout((2 * self.union_words,)),
        )
        s_keys = cute.make_tensor(
            cute.recast_ptr(s_union.iterator, dtype=self.raw_dtype),
            cute.make_layout((TILE,)),
        )
        s_index = cute.make_tensor(
            cute.recast_ptr(s_union.iterator, dtype=self.index_dtype),
            cute.make_layout((TILE,)),
        )
        s_prefix = smem.allocate_tensor(
            Int32, cute.make_layout((DIGITS,)), byte_alignment=16
        )
        s_relative = smem.allocate_tensor(
            Int32, cute.make_layout((DIGITS,)), byte_alignment=16
        )
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((WARPS + 2,)), byte_alignment=16
        )

        # Upsweep: ProcessRegion over the row.
        UP_LANES = const_expr(self.up_lanes)
        UP_LPW = const_expr(self.up_lanes_per_warp)
        local = cute.make_rmem_tensor(UP_LPW * 4, Int32)
        local.fill(0)
        keys = cute.make_rmem_tensor(ITEMS, self.word)
        for i in cutlass.range_constexpr(UP_LANES):
            s_union[i * THREADS + tidx] = Int32(0)
        offset = Int32(0)
        while n - offset >= Int32(self.unroll * TILE):
            for _ in cutlass.range_constexpr(self.unroll):
                self.upsweep_tile(
                    mKeysIn, row, offset, keys, s_bytes, tidx, current_bit, pass_bits
                )
                offset = offset + Int32(TILE)
            cute.arch.barrier()
            self.unpack(s_bytes, local, lane, warp)
            cute.arch.barrier()
            for i in cutlass.range_constexpr(UP_LANES):
                s_union[i * THREADS + tidx] = Int32(0)
        while n - offset >= Int32(TILE):
            self.upsweep_tile(
                mKeysIn, row, offset, keys, s_bytes, tidx, current_bit, pass_bits
            )
            offset = offset + Int32(TILE)
        for j in cutlass.range(offset + tidx, n, THREADS):
            digit = self.digit(self.word(mKeysIn[row, j]), current_bit, pass_bits)
            counter = ((digit >> Int32(2)) * Int32(THREADS) + tidx) * Int32(4) + (
                digit & Int32(3)
            )
            s_bytes[counter] = s_bytes[counter] + cutlass.Uint8(1)
        cute.arch.barrier()
        self.unpack(s_bytes, local, lane, warp)
        cute.arch.barrier()
        # ExtractCounts: per-lane partial counts, summed per digit.
        for k in cutlass.range_constexpr(UP_LPW):
            counter_lane = Int32(k * WARPS) + warp
            if counter_lane < Int32(UP_LANES):
                for sub in cutlass.range_constexpr(4):
                    s_union[lane * DIGITS + counter_lane * 4 + sub] = local[k * 4 + sub]
        cute.arch.barrier()
        count = Int32(0)
        if tidx < Int32(DIGITS):
            for i in cutlass.range_constexpr(32):
                count = count + s_union[i * DIGITS + tidx]
        cute.arch.barrier()
        excl, _ = block_exclusive_prefix(count, s_scan, lane, warp, WARPS)
        # bin_offset is relative to the row. cub's short circuit compares the
        # batch-absolute offsets with the row length, so only a row that
        # starts its batch can take it.
        begin = row_base + row * n
        if tidx == Int32(0):
            s_scan[WARPS + 1] = Int32(1)
        cute.arch.barrier()
        if tidx < Int32(DIGITS):
            if begin + excl != Int32(0):
                if begin + excl != n:
                    s_scan[WARPS + 1] = Int32(0)
        cute.arch.barrier()
        bin_offset = excl
        if s_scan[WARPS + 1] != Int32(0):
            # AgentRadixSortDownsweep::Copy of keys and values.
            for start in cutlass.range(0, n, TILE):
                for u in cutlass.range_constexpr(ITEMS):
                    j = start + tidx + Int32(u * THREADS)
                    if j < n:
                        keys[u] = self.word(mKeysIn[row, j])
                cute.arch.barrier()
                for u in cutlass.range_constexpr(ITEMS):
                    j = start + tidx + Int32(u * THREADS)
                    if j < n:
                        mKeysOut[row, j] = self.raw_dtype(keys[u])
            for start in cutlass.range(0, n, TILE):
                index = cute.make_rmem_tensor(ITEMS, self.index_dtype)
                for u in cutlass.range_constexpr(ITEMS):
                    j = start + tidx + Int32(u * THREADS)
                    if j < n:
                        if const_expr(first):
                            index[u] = self.index_dtype(j)
                        else:
                            index[u] = mIdxIn[row, j]
                cute.arch.barrier()
                for u in cutlass.range_constexpr(ITEMS):
                    j = start + tidx + Int32(u * THREADS)
                    if j < n:
                        mIdxOut[row, j] = index[u]
        else:
            for start in cutlass.range(0, n, TILE):
                bin_offset = self.downsweep_tile(
                    mKeysIn,
                    mIdxIn,
                    mKeysOut,
                    mIdxOut,
                    row,
                    start,
                    bin_offset,
                    keys,
                    s_union,
                    s_halves,
                    s_keys,
                    s_index,
                    s_prefix,
                    s_relative,
                    s_scan,
                    tidx,
                    lane,
                    warp,
                    current_bit,
                    pass_bits,
                    first,
                )
                cute.arch.barrier()

    @cute.jit
    def upsweep_tile(
        self,
        mKeysIn: cute.Tensor,
        row: Int32,
        offset: Int32,
        keys: cute.Tensor,
        s_bytes: cute.Tensor,
        tidx: Int32,
        current_bit: cutlass.Constexpr,
        pass_bits: cutlass.Constexpr,
    ):
        """ProcessFullTile: striped load, then one 8-bit counter bump per key."""
        for u in cutlass.range_constexpr(self.items):
            keys[u] = self.word(mKeysIn[row, offset + tidx + Int32(u * self.threads)])
        cute.arch.barrier()
        for u in cutlass.range_constexpr(self.items):
            digit = self.digit(keys[u], current_bit, pass_bits)
            counter = ((digit >> Int32(2)) * Int32(self.threads) + tidx) * Int32(4) + (
                digit & Int32(3)
            )
            s_bytes[counter] = s_bytes[counter] + cutlass.Uint8(1)

    @cute.jit
    def unpack(
        self, s_bytes: cute.Tensor, local: cute.Tensor, lane: Int32, warp: Int32
    ):
        """UnpackDigitCounts: warp w sums counter lanes w, w + WARPS, ..."""
        for k in cutlass.range_constexpr(self.up_lanes_per_warp):
            counter_lane = Int32(k * self.warps) + warp
            if counter_lane < Int32(self.up_lanes):
                for t in cutlass.range_constexpr(0, self.threads, 32):
                    word = (
                        counter_lane * Int32(self.threads) + lane + Int32(t)
                    ) * Int32(4)
                    for sub in cutlass.range_constexpr(4):
                        local[k * 4 + sub] = local[k * 4 + sub] + (
                            Int32(s_bytes[word + sub]) & Int32(0xFF)
                        )

    @cute.jit
    def downsweep_tile(
        self,
        mKeysIn: cute.Tensor,
        mIdxIn: cute.Tensor,
        mKeysOut: cute.Tensor,
        mIdxOut: cute.Tensor,
        row: Int32,
        start: Int32,
        bin_offset: Int32,
        keys: cute.Tensor,
        s_union: cute.Tensor,
        s_halves: cute.Tensor,
        s_keys: cute.Tensor,
        s_index: cute.Tensor,
        s_prefix: cute.Tensor,
        s_relative: cute.Tensor,
        s_scan: cute.Tensor,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
        current_bit: cutlass.Constexpr,
        pass_bits: cutlass.Constexpr,
        first: cutlass.Constexpr,
    ) -> Int32:
        """ProcessTile: transpose load, BlockRadixRank (memoize), scatter."""
        n = mKeysIn.shape[1]
        THREADS = const_expr(self.threads)
        ITEMS = const_expr(self.items)
        DIGITS = const_expr(self.digits)
        LANES = const_expr(self.rank_lanes)
        RAKING = const_expr(self.raking)
        valid = n - start
        # BLOCK_LOAD_TRANSPOSE: striped load, exchanged to a blocked arrangement.
        for u in cutlass.range_constexpr(ITEMS):
            j = tidx + Int32(u * THREADS)
            key = self.word(self.pad)
            if j < valid:
                key = self.word(mKeysIn[row, start + j])
            s_keys[j] = self.raw_dtype(key)
        cute.arch.barrier()
        for u in cutlass.range_constexpr(ITEMS):
            keys[u] = self.word(s_keys[tidx * Int32(ITEMS) + Int32(u)])
        cute.arch.barrier()

        # BlockRadixRank::RankKeys with 16-bit counters [lane][thread][2].
        for i in cutlass.range_constexpr(RAKING):
            s_union[i * THREADS + tidx] = Int32(0)
        ranks = cute.make_rmem_tensor(ITEMS, Int32)
        for u in cutlass.range_constexpr(ITEMS):
            digit = self.digit(keys[u], current_bit, pass_bits)
            counter = ((digit & Int32(LANES - 1)) * Int32(THREADS) + tidx) * Int32(
                2
            ) + (digit >> Int32(self.radix_bits - 1))
            ranks[u] = Int32(s_halves[counter])
            s_halves[counter] = cutlass.Uint16(ranks[u] + Int32(1))
        cute.arch.barrier()
        # ScanCounters: memoized raking segment, block scan, downsweep.
        segment = cute.make_rmem_tensor(RAKING, Int32)
        partial = Int32(0)
        for i in cutlass.range_constexpr(RAKING):
            segment[i] = s_union[tidx * Int32(RAKING) + Int32(i)]
            partial = partial + segment[i]
        running, total = block_exclusive_prefix(partial, s_scan, lane, warp, self.warps)
        running = running + (total << Int32(16))
        for i in cutlass.range_constexpr(RAKING):
            s_union[tidx * Int32(RAKING) + Int32(i)] = running
            running = running + segment[i]
        cute.arch.barrier()
        for u in cutlass.range_constexpr(ITEMS):
            digit = self.digit(keys[u], current_bit, pass_bits)
            counter = ((digit & Int32(LANES - 1)) * Int32(THREADS) + tidx) * Int32(
                2
            ) + (digit >> Int32(self.radix_bits - 1))
            ranks[u] = ranks[u] + Int32(s_halves[counter])
        excl = Int32(0)
        if tidx < Int32(DIGITS):
            excl = Int32(
                s_halves[
                    (tidx & Int32(LANES - 1)) * Int32(THREADS * 2)
                    + (tidx >> Int32(self.radix_bits - 1))
                ]
            )
        cute.arch.barrier()
        if tidx < Int32(DIGITS):
            s_prefix[tidx] = excl
        cute.arch.barrier()
        inclusive = Int32(self.tile)
        if tidx < Int32(DIGITS - 1):
            inclusive = s_prefix[tidx + Int32(1)]
        cute.arch.barrier()
        if tidx < Int32(DIGITS):
            bin_offset = bin_offset - excl
            s_relative[tidx] = bin_offset
            bin_offset = bin_offset + inclusive
        cute.arch.barrier()

        # ScatterKeys through shared memory, keeping each slot's offset.
        for u in cutlass.range_constexpr(ITEMS):
            s_keys[ranks[u]] = self.raw_dtype(keys[u])
        cute.arch.barrier()
        relative = cute.make_rmem_tensor(ITEMS, Int32)
        for u in cutlass.range_constexpr(ITEMS):
            i = tidx + Int32(u * THREADS)
            key = s_keys[i]
            relative[u] = s_relative[self.digit(self.word(key), current_bit, pass_bits)]
            if i < valid:
                mKeysOut[row, relative[u] + i] = key
        # GatherScatterValues: transpose load (or generate), then scatter.
        index = cute.make_rmem_tensor(ITEMS, self.index_dtype)
        cute.arch.barrier()
        if const_expr(first):
            for u in cutlass.range_constexpr(ITEMS):
                index[u] = self.index_dtype(start + tidx * Int32(ITEMS) + Int32(u))
        else:
            for u in cutlass.range_constexpr(ITEMS):
                j = tidx + Int32(u * THREADS)
                if j < valid:
                    s_index[j] = mIdxIn[row, start + j]
            cute.arch.barrier()
            for u in cutlass.range_constexpr(ITEMS):
                index[u] = s_index[tidx * Int32(ITEMS) + Int32(u)]
            cute.arch.barrier()
        cute.arch.barrier()
        for u in cutlass.range_constexpr(ITEMS):
            s_index[ranks[u]] = index[u]
        cute.arch.barrier()
        for u in cutlass.range_constexpr(ITEMS):
            i = tidx + Int32(u * THREADS)
            if i < valid:
                mIdxOut[row, relative[u] + i] = s_index[i]
        return bin_offset


class SegmentedRadixSort:
    """All passes of DispatchSegmentedRadixSort for one chunk of rows."""

    def __init__(
        self,
        kind: str,
        key_bits: int,
        index_dtype: type[cutlass.Numeric],
        descending: bool,
        arch: int,
    ):
        policy, alt_policy = segmented_policies(
            arch, key_bits // 8, index_dtype.width // 8
        )
        self.primary = SegmentedPass(kind, key_bits, index_dtype, descending, policy)
        self.alt = SegmentedPass(kind, key_bits, index_dtype, descending, alt_policy)
        self.schedule = pass_schedule(key_bits, policy[2], alt_policy[2])

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mKeysTmp: cute.Tensor,
        mIdxTmp: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        row_base: Int32,
        stream: cuda.CUstream,
    ):
        rows = mX.shape[0]
        PASSES = const_expr(len(self.schedule))
        # The last pass writes the outputs and earlier ones alternate with the
        # temporaries, as cub does when it may not overwrite its input.
        keys = [mX]
        index = [mIdxTmp]
        for p in cutlass.range_constexpr(PASSES):
            current_bit, pass_bits, alt = self.schedule[p]
            to_out = (PASSES - 1 - p) % 2 == 0
            keys.append(mValues if to_out else mKeysTmp)
            index.append(mIndices if to_out else mIdxTmp)
            stage = self.alt if alt else self.primary
            # Two CTAs per SM cap the registers; the 192 x 39 policy would
            # otherwise take 255 and run one CTA per SM.
            stage.kernel(
                keys[p],
                index[p],
                keys[p + 1],
                index[p + 1],
                row_base,
                current_bit,
                pass_bits,
                p == 0,
            ).launch(
                grid=[rows, 1, 1],
                block=[stage.threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=2,
            )


def spec_name(spec: dict) -> str:
    return (
        f"sort_segmented_radix_{spec['kind']}{spec['key_bits']}_{spec['index']}"
        f"_{'desc' if spec['descending'] else 'asc'}_sm{spec['arch']}"
    )


def make_spec(
    arch: int, key_dtype: torch.dtype, index_dtype: torch.dtype, descending: bool
) -> dict:
    kind, key_bits = KEY_KINDS[key_dtype]
    return {
        "kernel": "segmented_radix",
        "kind": kind,
        "key_bits": key_bits,
        "index": index_name(index_dtype),
        "descending": descending,
        "arch": arch,
    }


def compile_args(spec: dict) -> tuple[SegmentedRadixSort, list]:
    """Kernel object and fake operands for one spec; shared by JIT and AOT."""
    raw = RAW_CUTE[spec["key_bits"]]
    index = INDEX_DTYPES[getattr(torch, spec["index"])]
    kernel = SegmentedRadixSort(
        spec["kind"], spec["key_bits"], index, spec["descending"], spec["arch"]
    )
    fake = [
        fake_rows(raw),
        fake_rows(raw),
        fake_rows(index),
        fake_rows(raw),
        fake_rows(index),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return kernel, fake


@instrumented_cutedsl_cache("aten::sort", key_fn=lambda spec: spec_name(dict(spec)))
def _compile(spec: tuple):
    kernel, fake = compile_args(dict(spec))
    return compile_kernel(kernel, *fake)


def segmented_sort(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    descending: bool,
    arch: int,
) -> None:
    """Sort each row of one stock batch (2D row-major ``x``) into ``values`` and
    ``indices``; the caller checks eligibility as for the other leaves."""
    spec = make_spec(arch, x.dtype, indices.dtype, descending)
    kernel = _compile(tuple(sorted(spec.items())))
    rows, n = x.shape
    if rows == 0 or n == 0:
        return
    chunk = min(rows, max(_MIN_CHUNK_ROWS, CHUNK_ITEMS // n))
    key_bits = KEY_KINDS[x.dtype][1]
    keys_tmp = torch.empty((chunk, n), dtype=RAW_TORCH[key_bits], device=x.device)
    idx_tmp = torch.empty((chunk, n), dtype=indices.dtype, device=x.device)
    x_raw, values_raw = raw_view(x), raw_view(values)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        kernel(
            read_only(x_raw[start:stop]),
            keys_tmp[: stop - start],
            idx_tmp[: stop - start],
            values_raw[start:stop],
            indices[start:stop],
            start * n,
        )
