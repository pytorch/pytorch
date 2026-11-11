"""Device-wide radix leaves: mirror the cub sorts of ``launch_stable_sort_kernel``.

``OnesweepSort`` ports cub's DeviceRadixSort onesweep path for one array
(dispatch_radix_sort.cuh InvokeOnesweep): a histogram kernel counts the digits
of every 8-bit pass, an exclusive-sum kernel turns them into offsets, then per
pass and per portion one onesweep kernel ranks a tile per CTA with
RADIX_RANK_MATCH_EARLY_COUNTS_ANY and finds its global offsets with a decoupled
look-back (agent_radix_sort_onesweep.cuh). It serves two leaves:

* one row, or rows of a million elements and more: one sort per row
  (segmented_sort_large_segments);
* fewer than 128 rows: segmented_sort_pairs_by_full_sort, i.e. a full sort of
  the keys carrying each element's position, a keys-only sort of the positions
  by segment, and the postprocess gather.

Deviations from the stock code: the first pass generates the payload instead of
reading an iota or (segment, index) buffer; the full sort carries the flat
position (int32) instead of an int2 (segment, index) pair; values and narrow
indices are written straight to the outputs.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch
from torch._native.cutedsl.hw_caps import caps
from torch._native.cutedsl.launch import compile_kernel, fake_compact, read_only, sym
from torch._native.cutedsl.scan import block_exclusive_prefix
from torch._native.instrumentation import instrumented_cutedsl_cache

from .cutedsl_common import (
    INDEX_DTYPES,
    index_name,
    KEY_KINDS,
    match_any,
    RADIX_BINS,
    RADIX_BITS,
    radix_ordinal,
    RAW_CUTE,
    RAW_TORCH,
    raw_view,
)
from .policy import histogram_policy, onesweep_policy, stock_leaf


# ExclusiveSumPolicy threads and the onesweep portion bound of cub's dispatch.
_SCAN_THREADS = 256
_PORTION_LIMIT = (1 << 28) - 1
_LOOKBACK_PARTIAL = 1 << 30
_LOOKBACK_GLOBAL = -(1 << 31)
_LOOKBACK_VALUE = (1 << 30) - 1
_INT_MAX = (1 << 31) - 1


class OnesweepSort:
    """One cub onesweep sort of a 1D key array.

    ``kind == "segment"`` is the keys-only sort of flat positions by
    ``position // nsort`` over ``segment_bits`` bits (the int64 (segment,
    index) sort of the full-sort leaf); otherwise keys are raw bits of
    ``key_bits`` sorted over all bits in 8-bit passes, with a payload that the
    first pass generates as the element position.
    """

    def __init__(
        self,
        kind: str,
        key_bits: int,
        payload: type[cutlass.Numeric] | None,
        descending: bool,
        arch: int,
    ):
        self.kind = kind
        self.key_bits = key_bits
        self.raw_dtype = Int32 if kind == "segment" else RAW_CUTE[key_bits]
        self.word = Int64 if key_bits == 64 else Int32
        self.keys_only = payload is None
        self.payload = payload or Int32
        self.descending = descending
        self.passes = 1 if kind == "segment" else key_bits // RADIX_BITS
        self.threads, self.items = onesweep_policy(
            arch, key_bits // 8, 0 if self.keys_only else self.payload.width // 8
        )
        self.tile = self.threads * self.items
        self.portion = _PORTION_LIMIT // self.tile * self.tile
        self.hist_threads, self.hist_items = histogram_policy(arch)
        # Raw bits of the largest ordinal: cub's padding key.
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
        widths = [key_bits] + ([] if self.keys_only else [self.payload.width])
        self.union_words = max(
            self.tile * max(widths) // 32 + 1, self.threads // 32 * RADIX_BINS
        )

    def state_words(self, num_items: int) -> int:
        """Counters and bins per (portion, pass), then one portion's look-back."""
        portions = max(1, -(-num_items // self.portion))
        blocks = -(-min(num_items, self.portion) // self.tile)
        return portions * self.passes * (1 + RADIX_BINS) + blocks * RADIX_BINS

    @cute.jit
    def __call__(
        self,
        mKeys: cute.Tensor,
        mKeysTmp: cute.Tensor,
        mKeysOut: cute.Tensor,
        mPayTmp: cute.Tensor,
        mPayOut: cute.Tensor,
        mState: cute.Tensor,
        hist_blocks: Int32,
        nsort: Int32,
        segment_bits: Int32,
        stream: cuda.CUstream,
    ):
        num_items = mKeys.shape[0]
        PASSES = const_expr(self.passes)
        portions = (num_items - Int32(1)) // Int32(self.portion) + Int32(1)
        bins = portions * Int32(PASSES)
        lookback = bins * Int32(1 + RADIX_BINS)
        # cub clears the tile counters and the first portion's bins.
        self.clear(mState, Int32(0), bins + Int32(PASSES * RADIX_BINS)).launch(
            grid=[(bins + Int32(PASSES * RADIX_BINS) + 255) // 256, 1, 1],
            block=[256, 1, 1],
            stream=stream,
        )
        self.histogram(mKeys, mState, bins, nsort, segment_bits).launch(
            grid=[hist_blocks, 1, 1], block=[self.hist_threads, 1, 1], stream=stream
        )
        self.exclusive_sum(mState, bins).launch(
            grid=[PASSES, 1, 1], block=[_SCAN_THREADS, 1, 1], stream=stream
        )
        # The last pass writes the outputs and earlier ones alternate with the
        # temporaries, as cub does when it may not overwrite its input.
        keys = [mKeys]
        payload = [mPayTmp]
        for p in cutlass.range_constexpr(PASSES):
            to_out = (PASSES - 1 - p) % 2 == 0
            keys.append(mKeysOut if to_out else mKeysTmp)
            payload.append(mPayOut if to_out else mPayTmp)
            for portion in cutlass.range(portions):
                items = cutlass.min(
                    num_items - portion * Int32(self.portion), Int32(self.portion)
                )
                blocks = (items + Int32(self.tile - 1)) // Int32(self.tile)
                self.clear(mState, lookback, blocks * Int32(RADIX_BINS)).launch(
                    grid=[blocks, 1, 1], block=[RADIX_BINS, 1, 1], stream=stream
                )
                self.onesweep(
                    keys[p],
                    payload[p],
                    keys[p + 1],
                    payload[p + 1],
                    mState,
                    Int32(p),
                    portion,
                    portions,
                    items,
                    nsort,
                    segment_bits,
                ).launch(
                    grid=[blocks, 1, 1],
                    block=[self.threads, 1, 1],
                    stream=stream,
                    min_blocks_per_mp=2,
                )

    @cute.kernel
    def clear(self, mState: cute.Tensor, start: Int32, count: Int32):
        """Stands in for cub's cudaMemsetAsync."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        i = bidx * bdim + tidx
        if i < count:
            mState[start + i] = Int32(0)

    @cute.jit
    def digit(
        self, key, current_bit: Int32, nsort: Int32, segment_bits: Int32
    ) -> Int32:
        if const_expr(self.kind == "segment"):
            return (key // nsort) & ((Int32(1) << segment_bits) - Int32(1))
        ord_ = radix_ordinal(key, self.kind, self.key_bits, self.descending)
        return Int32((ord_ >> self.word(current_bit)) & self.word(RADIX_BINS - 1))

    @cute.jit
    def pad_key(self, nsort: Int32, segment_bits: Int32):
        if const_expr(self.kind == "segment"):
            return ((Int32(1) << segment_bits) - Int32(1)) * nsort
        return self.word(self.pad)

    @cute.kernel
    def histogram(
        self,
        mKeys: cute.Tensor,
        mState: cute.Tensor,
        bins: Int32,
        nsort: Int32,
        segment_bits: Int32,
    ):
        """AgentRadixSortHistogram: grid-strided tiles into shared counters."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        num_items = mKeys.shape[0]
        PASSES = const_expr(self.passes)
        THREADS = const_expr(self.hist_threads)
        ITEMS = const_expr(self.hist_items)
        smem = cutlass.utils.SmemAllocator()
        s_bins = smem.allocate_tensor(
            Int32, cute.make_layout((PASSES * RADIX_BINS,)), byte_alignment=16
        )
        for i in cutlass.range(tidx, PASSES * RADIX_BINS, THREADS):
            s_bins[i] = Int32(0)
        cute.arch.barrier()
        keys = cute.make_rmem_tensor(ITEMS, self.word)
        pad = self.pad_key(nsort, segment_bits)
        # Loop over tile indices: an element-offset counter would pass 2^31 - 1
        # (and wrap) on rows within one grid step of it.
        tiles = (num_items - Int32(1)) // Int32(THREADS * ITEMS) + Int32(1)
        for t in cutlass.range(bidx, tiles, grid):
            tile = t * Int32(THREADS * ITEMS)
            for u in cutlass.range_constexpr(ITEMS):
                j = tile + Int32(u * THREADS) + tidx
                keys[u] = pad
                if j < num_items:
                    keys[u] = self.word(mKeys[j])
            for p in cutlass.range_constexpr(PASSES):
                for u in cutlass.range_constexpr(ITEMS):
                    digit = self.digit(
                        keys[u], Int32(p * RADIX_BITS), nsort, segment_bits
                    )
                    cute.arch.atomic_add(
                        s_bins.iterator + Int32(p * RADIX_BINS) + digit, Int32(1)
                    )
        cute.arch.barrier()
        for i in cutlass.range(tidx, PASSES * RADIX_BINS, THREADS):
            count = s_bins[i]
            if count > Int32(0):
                cute.arch.atomic_add(mState.iterator + bins + i, count)

    @cute.kernel
    def exclusive_sum(self, mState: cute.Tensor, bins: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        p, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((_SCAN_THREADS // 32 + 1,)), byte_alignment=16
        )
        i = bins + p * Int32(RADIX_BINS) + tidx
        excl, _ = block_exclusive_prefix(
            mState[i], s_scan, tidx % Int32(32), tidx // Int32(32), _SCAN_THREADS // 32
        )
        mState[i] = excl

    @cute.kernel
    def onesweep(
        self,
        mKeysIn: cute.Tensor,
        mPayIn: cute.Tensor,
        mKeysOut: cute.Tensor,
        mPayOut: cute.Tensor,
        mState: cute.Tensor,
        pass_: Int32,
        portion: Int32,
        portions: Int32,
        num_items: Int32,
        nsort: Int32,
        segment_bits: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        THREADS = const_expr(self.threads)
        ITEMS = const_expr(self.items)
        TILE = const_expr(self.tile)
        NUM_WARPS = const_expr(self.threads // 32)
        PASSES = const_expr(self.passes)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        current_bit = pass_ * Int32(RADIX_BITS)
        base = portion * Int32(self.portion)
        counter = portion * Int32(PASSES) + pass_
        bins_in = portions * Int32(PASSES) + counter * Int32(RADIX_BINS)
        bins_out = bins_in + Int32(PASSES * RADIX_BINS)
        lookback = portions * Int32(PASSES * (1 + RADIX_BINS))

        smem = cutlass.utils.SmemAllocator()
        # cub's union of the rank counters and the tile's keys or values.
        s_union = smem.allocate_tensor(
            Int32, cute.make_layout((self.union_words,)), byte_alignment=16
        )
        s_keys = cute.make_tensor(
            cute.recast_ptr(s_union.iterator, dtype=self.raw_dtype),
            cute.make_layout((TILE,)),
        )
        s_offset = smem.allocate_tensor(
            Int32, cute.make_layout((RADIX_BINS,)), byte_alignment=16
        )
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((NUM_WARPS + 3,)), byte_alignment=16
        )

        # Tiles are numbered in start order so look-back only waits on running CTAs.
        if tidx == Int32(0):
            s_scan[NUM_WARPS + 1] = cute.arch.atomic_add(
                mState.iterator + counter, Int32(1)
            )
            s_scan[NUM_WARPS + 2] = Int32(0)
        cute.arch.barrier()
        block = s_scan[NUM_WARPS + 1]
        tile_start = base + block * Int32(TILE)
        tile_items = cutlass.min(num_items - block * Int32(TILE), Int32(TILE))
        # UpdateBinsGlobal: the last tile of a portion seeds the next one.
        carry = Int32(0)
        if (block + Int32(1)) * Int32(TILE) >= num_items:
            if portion < portions - Int32(1):
                carry = Int32(1)

        keys = cute.make_rmem_tensor(ITEMS, self.word)
        rank = cute.make_rmem_tensor(ITEMS, Int32)
        warp_base = warp * Int32(32 * ITEMS) + lane
        pad = self.pad_key(nsort, segment_bits)
        for u in cutlass.range_constexpr(ITEMS):
            j = warp_base + Int32(u * 32)
            keys[u] = pad
            if j < tile_items:
                keys[u] = self.word(mKeysIn[tile_start + j])

        # BlockRadixRankMatchEarlyCounts: per-warp digit histograms.
        for i in cutlass.range_constexpr(RADIX_BINS // 32):
            s_union[warp * RADIX_BINS + lane + i * 32] = Int32(0)
        cute.arch.sync_warp()
        for u in cutlass.range_constexpr(ITEMS):
            digit = self.digit(keys[u], current_bit, nsort, segment_bits)
            cute.arch.atomic_add(s_union.iterator + warp * RADIX_BINS + digit, Int32(1))
        cute.arch.barrier()
        # Thread b owns digit b: warp counts -> warp offsets, then the counts
        # callback publishes the tile's count and checks for a single digit.
        count = Int32(0)
        lb = lookback + block * Int32(RADIX_BINS) + tidx
        if tidx < Int32(RADIX_BINS):
            for w in cutlass.range_constexpr(NUM_WARPS):
                c = s_union[w * RADIX_BINS + tidx]
                s_union[w * RADIX_BINS + tidx] = count
                count = count + c
            cute.arch.store(
                mState.iterator + lb,
                count | Int32(_LOOKBACK_PARTIAL),
                sem="relaxed",
                scope="gpu",
            )
            if count == Int32(TILE):
                s_scan[NUM_WARPS + 2] = Int32(1)
        excl, _ = block_exclusive_prefix(count, s_scan, lane, warp, NUM_WARPS)
        if s_scan[NUM_WARPS + 2] != Int32(0):
            # Short circuit: the whole tile has one digit and is copied to
            # that digit's run.
            common = self.digit(keys[0], current_bit, nsort, segment_bits)
            if tidx < Int32(RADIX_BINS):
                excl = Int32(0)
                if tidx > common:
                    excl = Int32(TILE)
                self.lookback(mState, lb, block, count, excl, bins_in, s_offset, tidx)
                if carry != Int32(0):
                    mState[bins_out + tidx] = s_offset[tidx] + excl + count
            cute.arch.barrier()
            first = s_offset[common]
            for u in cutlass.range_constexpr(ITEMS):
                j = warp_base + Int32(u * 32)
                if j < tile_items:
                    mKeysOut[first + j] = self.raw_dtype(keys[u])
            if const_expr(not self.keys_only):
                for u in cutlass.range_constexpr(ITEMS):
                    j = warp_base + Int32(u * 32)
                    if j < tile_items:
                        value = self.payload(tile_start + j)
                        if pass_ != Int32(0):
                            value = mPayIn[tile_start + j]
                        mPayOut[first + j] = value
        else:
            if tidx < Int32(RADIX_BINS):
                for w in cutlass.range_constexpr(NUM_WARPS):
                    s_union[w * RADIX_BINS + tidx] = (
                        s_union[w * RADIX_BINS + tidx] + excl
                    )
            cute.arch.barrier()
            lanes_le = Int32(cute.arch.lanemask_le())
            for u in cutlass.range_constexpr(ITEMS):
                digit = self.digit(keys[u], current_bit, nsort, segment_bits)
                peers = match_any(digit, RADIX_BITS)
                leader = Int32(31) - Int32(cute.arch.clz(peers))
                below = cute.arch.popc(peers & lanes_le)
                offset = Int32(0)
                if lane == leader:
                    offset = cute.arch.atomic_add(
                        s_union.iterator + warp * RADIX_BINS + digit, below
                    )
                offset = cute.arch.shuffle_sync(offset, leader)
                rank[u] = offset + below - Int32(1)
            cute.arch.barrier()

            for u in cutlass.range_constexpr(ITEMS):
                s_keys[rank[u]] = self.raw_dtype(keys[u])
            if tidx < Int32(RADIX_BINS):
                self.lookback(mState, lb, block, count, excl, bins_in, s_offset, tidx)
                if carry != Int32(0):
                    mState[bins_out + tidx] = s_offset[tidx] + excl + count
            cute.arch.barrier()

            # Keep the digit of slot tidx + u * THREADS in the high half of
            # rank[u] for the value scatter, which overwrites the keys.
            for u in cutlass.range_constexpr(ITEMS):
                i = tidx + Int32(u * THREADS)
                key = s_keys[i]
                digit = self.digit(self.word(key), current_bit, nsort, segment_bits)
                rank[u] = rank[u] | (digit << Int32(16))
                if i < tile_items:
                    mKeysOut[s_offset[digit] + i] = key
            if const_expr(not self.keys_only):
                s_values = cute.make_tensor(
                    cute.recast_ptr(s_union.iterator, dtype=self.payload),
                    cute.make_layout((TILE,)),
                )
                values = cute.make_rmem_tensor(ITEMS, self.payload)
                for u in cutlass.range_constexpr(ITEMS):
                    j = warp_base + Int32(u * 32)
                    values[u] = self.payload(tile_start + j)
                    if pass_ != Int32(0):
                        if j < tile_items:
                            values[u] = mPayIn[tile_start + j]
                cute.arch.barrier()
                for u in cutlass.range_constexpr(ITEMS):
                    s_values[rank[u] & Int32(0xFFFF)] = values[u]
                cute.arch.barrier()
                for u in cutlass.range_constexpr(ITEMS):
                    i = tidx + Int32(u * THREADS)
                    if i < tile_items:
                        mPayOut[s_offset[rank[u] >> Int32(16)] + i] = s_values[i]

    @cute.jit
    def lookback(
        self,
        mState: cute.Tensor,
        lb: Int32,
        block: Int32,
        count: Int32,
        excl: Int32,
        bins_in: Int32,
        s_offset: cute.Tensor,
        tidx: Int32,
    ):
        """LoadBinsToOffsetsGlobal and LookbackGlobal for digit ``tidx``."""
        inclusive = count
        prev = block
        while prev > Int32(0):
            prev = prev - Int32(1)
            status = Int32(0)
            while status == Int32(0):
                status = cute.arch.load(
                    mState.iterator + lb - (block - prev) * Int32(RADIX_BINS),
                    Int32,
                    sem="relaxed",
                    scope="gpu",
                )
            inclusive = inclusive + (status & Int32(_LOOKBACK_VALUE))
            if status < Int32(0):
                prev = Int32(0)
        cute.arch.store(
            mState.iterator + lb,
            inclusive | Int32(_LOOKBACK_GLOBAL),
            sem="relaxed",
            scope="gpu",
        )
        s_offset[tidx] = mState[bins_in + tidx] - excl + inclusive - count


class FullSortPostprocess:
    """sort_postprocess_kernel: gather the sorted positions back into rows."""

    def __init__(self, index_dtype: type[cutlass.Numeric]):
        self.index_dtype = index_dtype

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        mPositions: cute.Tensor,
        nsort: Int32,
        stream: cuda.CUstream,
    ):
        n = mX.shape[0]
        self.kernel(mX, mValues, mIndices, mPositions, nsort).launch(
            grid=[(n - 1) // 512 + 1, 1, 1], block=[512, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        mPositions: cute.Tensor,
        nsort: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * Int32(512) + tidx
        if i < mX.shape[0]:
            first = i // nsort * nsort
            idx = mPositions[i] - first
            mIndices[i] = self.index_dtype(idx)
            mValues[i] = mX[first + idx]


def onesweep_spec(
    arch: int, kind: str, key_bits: int, payload: str | None, descending: bool
) -> dict:
    return {
        "kernel": "onesweep",
        "kind": kind,
        "key_bits": key_bits,
        "payload": payload,
        "descending": descending,
        "arch": arch,
    }


def spec_name(spec: dict) -> str:
    if spec["kernel"] == "full_sort_postprocess":
        return f"sort_full_sort_postprocess_{spec['key_bits']}_{spec['index']}"
    return (
        f"sort_onesweep_{spec['kind']}{spec['key_bits']}_{spec['payload']}"
        f"_{'desc' if spec['descending'] else 'asc'}_sm{spec['arch']}"
    )


def _sorter(spec: dict) -> OnesweepSort:
    payload = spec["payload"] and INDEX_DTYPES[getattr(torch, spec["payload"])]
    return OnesweepSort(
        spec["kind"], spec["key_bits"], payload, spec["descending"], spec["arch"]
    )


def compile_args(spec: dict) -> tuple[OnesweepSort | FullSortPostprocess, list]:
    """Kernel object and fake operands for one spec; shared by JIT and AOT."""
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if spec["kernel"] == "full_sort_postprocess":
        raw = RAW_CUTE[spec["key_bits"]]
        index = INDEX_DTYPES[getattr(torch, spec["index"])]
        kernel = FullSortPostprocess(index)
        fake = [fake_compact(t, (sym(),)) for t in (raw, raw, index, Int32)]
        return kernel, [*fake, Int32(1), stream]
    kernel = _sorter(spec)
    raw, pay = kernel.raw_dtype, kernel.payload
    fake = [fake_compact(t, (sym(),)) for t in (raw, raw, raw, pay, pay, Int32)]
    return kernel, [*fake, Int32(1), Int32(1), Int32(1), stream]


@instrumented_cutedsl_cache("aten::sort", key_fn=lambda spec: spec_name(dict(spec)))
def _compile(spec: tuple):
    kernel, fake = compile_args(dict(spec))
    return compile_kernel(kernel, *fake)


def _run(spec: dict, *args) -> None:
    _compile(tuple(sorted(spec.items())))(*args)


def _hist_blocks(device: torch.device, arch: int) -> int:
    # cub sizes the histogram grid by occupancy: SMs times blocks per SM.
    return caps(device).fill_blocks(histogram_policy(arch)[0])


def _per_row(x, values, indices, descending: bool, arch: int) -> None:
    """segmented_sort_large_segments: one onesweep sort per row."""
    rows, n = x.shape
    kind, key_bits = KEY_KINDS[x.dtype]
    spec = onesweep_spec(arch, kind, key_bits, index_name(indices.dtype), descending)
    sorter = _sorter(spec)
    state = torch.empty(sorter.state_words(n), dtype=torch.int32, device=x.device)
    x_raw, values_raw = raw_view(x), raw_view(values)
    keys_tmp, idx_tmp = values_raw[0], indices[0]
    if sorter.passes > 1:
        keys_tmp = torch.empty(n, dtype=RAW_TORCH[key_bits], device=x.device)
        idx_tmp = torch.empty(n, dtype=indices.dtype, device=x.device)
    blocks = _hist_blocks(x.device, arch)
    for r in range(rows):
        out_keys, out_idx = values_raw[r], indices[r]
        if sorter.passes == 1:
            keys_tmp, idx_tmp = out_keys, out_idx
        _run(
            spec,
            read_only(x_raw[r]),
            keys_tmp,
            out_keys,
            idx_tmp,
            out_idx,
            state,
            blocks,
            n,
            0,
        )


def _full_sort(x, values, indices, descending: bool, arch: int) -> None:
    """segmented_sort_pairs_by_full_sort on fewer than 128 rows."""
    rows, nsort = x.shape
    n = rows * nsort
    kind, key_bits = KEY_KINDS[x.dtype]
    pairs = onesweep_spec(arch, kind, key_bits, "int32", descending)
    by_segment = onesweep_spec(arch, "segment", 32, None, False)
    post = {
        "kernel": "full_sort_postprocess",
        "key_bits": key_bits,
        "index": index_name(indices.dtype),
    }
    words = max(_sorter(s).state_words(n) for s in (pairs, by_segment))
    state = torch.empty(words, dtype=torch.int32, device=x.device)
    positions = torch.empty(n, dtype=torch.int32, device=x.device)
    positions2 = torch.empty(n, dtype=torch.int32, device=x.device)
    x_flat = raw_view(x).view(-1)
    values_flat = raw_view(values).view(-1)
    # The pairs sort's key output is discarded, so it lands in values.
    keys_tmp = values_flat
    if key_bits > RADIX_BITS:
        keys_tmp = torch.empty(n, dtype=RAW_TORCH[key_bits], device=x.device)
    blocks = _hist_blocks(x.device, arch)
    _run(
        pairs,
        read_only(x_flat),
        keys_tmp,
        values_flat,
        positions,
        positions2,
        state,
        blocks,
        nsort,
        0,
    )
    # Sort on the low segment_bits bits of the (segment, index) pairs.
    segment_bits = max(1, (rows - 1).bit_length())
    _run(
        by_segment,
        positions2,
        positions2,
        positions,
        positions2,
        positions2,
        state,
        blocks,
        nsort,
        segment_bits,
    )
    _run(post, read_only(x_flat), values_flat, indices.view(-1), positions, nsort)


def global_sort(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    descending: bool,
    arch: int,
) -> None:
    """Sort each row of 2D row-major ``x`` into ``values`` and ``indices``.

    Walks the batches of ``launch_stable_sort_kernel`` and runs each batch's
    leaf: per-row onesweep, the full sort, or the segmented radix sort. The
    caller checks eligibility: key and index dtypes supported, contiguous
    outputs and rows that ``indices`` can address.
    """
    rows, nsort = x.shape
    if rows == 0 or nsort == 0:
        return
    batch = max(1, min(rows * nsort, _INT_MAX) // nsort)
    for start in range(0, rows, batch):
        stop = min(start + batch, rows)
        args = (x[start:stop], values[start:stop], indices[start:stop], descending)
        leaf = stock_leaf(nsort, stop - start)
        if leaf == "per_row":
            _per_row(*args, arch)
        elif leaf == "full_sort":
            _full_sort(*args, arch)
        elif leaf == "segmented":
            from .cutedsl_segmented import segmented_sort

            segmented_sort(*args, arch)
        else:
            raise ValueError(f"rows of {nsort} take the in-place leaf")
