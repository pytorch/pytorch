"""Partition leaf: the stable sort of bool keys as a counting partition.

The stable order of a bool row is unique: the keys that sort first (false
ascending, true descending) in input order, then the others in input order.
So ``values`` only needs each row's first-class count ``F`` and ``indices`` is
a two-way stream compaction; no key is compared and no iota is read.

Rows are read through ``vec``-byte chunks aligned in the flat buffer, so one
chunk can hold the end of a row and the start of the next; bytes outside the
row are masked. A tile's thread-contiguous chunks give every element its
exclusive first-class count from one block scan. Indices are staged in shared
memory in output order, each run at its gmem alignment, and leave as
``ivec``-wide stores; values are written from registers. Three modes:

- ``rows``: a CTA owns ``group`` complete rows in one tile. Per-row counts come
  from the in-tile prefix at each row start, so short rows need no extra pass.
- ``row``: a CTA owns a whole long row: it counts the row, then re-reads it
  (from L2) to scatter, tile by tile.
- ``split``: rows are cut into segments of ``group`` chunks over several CTAs; a
  count kernel writes per-segment counts and the scatter kernel prefixes them.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, Uint8

import torch
from torch._native.cutedsl.hw_caps import caps
from torch._native.cutedsl.launch import (
    compile_kernel,
    fake_compact,
    read_only,
    supported_alignment,
    sym,
)
from torch._native.cutedsl.scan import block_exclusive_prefix, warp_inclusive_prefix
from torch._native.instrumentation import instrumented_cutedsl_cache

from .cutedsl_common import INDEX_DTYPES, index_name


class PartitionSort:
    def __init__(
        self,
        index_dtype: type[cutlass.Numeric],
        descending: bool,
        mode: str,
        threads: int,
        items: int,
        vec: int,
        ivec: int,
        max_rows: int,
    ):
        self.index_dtype = index_dtype
        self.descending = descending
        self.mode = mode
        self.threads = threads
        self.items = items
        self.vec = vec
        self.ivec = ivec
        self.max_rows = max_rows
        self.tile = threads * items * vec

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        mCounts: cute.Tensor,
        rows: Int32,
        n: Int32,
        group: Int32,
        stream,
    ):
        block = [self.threads, 1, 1]
        if const_expr(self.mode == "rows"):
            self.rows_kernel(mX, mValues, mIndices, rows, n, group).launch(
                grid=[(rows + group - Int32(1)) // group, 1, 1],
                block=block,
                stream=stream,
            )
        else:
            segs = (n + Int32(2 * self.vec - 2)) // Int32(self.vec)
            segs = (segs + group - Int32(1)) // group
            if const_expr(self.mode == "split"):
                self.count_kernel(mX, mCounts, n, group, segs).launch(
                    grid=[rows * segs, 1, 1], block=block, stream=stream
                )
            self.row_kernel(mX, mValues, mIndices, mCounts, n, group, segs).launch(
                grid=[rows * segs, 1, 1], block=block, stream=stream
            )

    @cute.jit
    def chunked(self, t: cute.Tensor, width: cutlass.Constexpr) -> cute.Tensor:
        """``t`` as ``(width, chunks)``; chunk ``c`` keeps the base alignment."""
        chunks = (t.shape[0] + Int32(width - 1)) // Int32(width)
        return cute.make_tensor(
            t.iterator, cute.make_layout((width, chunks), stride=(1, width))
        )

    @cute.jit
    def load(
        self,
        gX: cute.Tensor,
        c0: Int64,
        q0: Int32,
        q1: Int32,
        hi: Int32,
        shift: Int32,
        chunks: cute.Tensor,
        masks: cute.Tensor,
    ) -> Int32:
        """Load this thread's chunks ``q0 + v < q1`` past chunk ``c0`` and set
        ``masks[v]`` to the first-class bits of bytes ``0 <= ol < hi``, where
        ``ol = vec * q + k - shift``; returns their count."""
        VEC = const_expr(self.vec)
        f = Int32(0)
        for v in cutlass.range_constexpr(self.items):
            masks[v] = Int32(0)
            q = q0 + Int32(v)
            if q < q1:
                cute.autovec_copy(gX[None, c0 + Int64(q)], chunks[None, v])
                base = q * Int32(VEC) - shift
                m = Int32(0)
                for k in cutlass.range_constexpr(VEC):
                    first = (chunks[k, v] != Uint8(0)) == self.descending
                    if first and base + Int32(k) >= Int32(0) and base + Int32(k) < hi:
                        m = m | Int32(1 << k)
                masks[v] = m
                f = f + cute.arch.popc(m)
        return f

    @cute.jit
    def store_values(
        self,
        mValues: cute.Tensor,
        gV: cute.Tensor,
        c0: Int64,
        q: Int32,
        base: Int32,
        lo: Int32,
        hi: Int32,
        out: cute.Tensor,
    ):
        VEC = const_expr(self.vec)
        if base >= lo and base + Int32(VEC) <= hi:
            cute.autovec_copy(out, gV[None, c0 + Int64(q)])
        else:
            for k in cutlass.range_constexpr(VEC):
                if base + Int32(k) >= lo and base + Int32(k) < hi:
                    mValues[(c0 + Int64(q)) * Int64(VEC) + Int64(k)] = out[k]

    @cute.jit
    def write_runs(
        self,
        mIndices: cute.Tensor,
        s_idx: cute.Tensor,
        p1: Int32,
        d1: Int64,
        len1: Int32,
        p2: Int32,
        d2: Int64,
        len2: Int32,
    ):
        """Copy the staged runs ``s_idx[p : p + len]`` to ``mIndices[d :]``.

        Each run is staged at ``p == d (mod ivec)`` and the second run starts
        on a fresh ``ivec`` slot, so every full slot is one aligned store."""
        E = const_expr(self.ivec)
        tidx, _, _ = cute.arch.thread_idx()
        gI = self.chunked(mIndices, E)
        sI = self.chunked(s_idx, E)
        frag = cute.make_rmem_tensor(E, self.index_dtype)
        split = (p2 // Int32(E)) * Int32(E)
        end = p2 + len2
        if len2 == Int32(0):
            split = p1 + len1 + Int32(E)
            end = p1 + len1
        for w in cutlass.range(
            p1 // Int32(E) + tidx, (end + Int32(E - 1)) // Int32(E), self.threads
        ):
            j0 = w * Int32(E)
            p, d, length = p1, d1, len1
            if j0 >= split:
                p, d, length = p2, d2, len2
            if j0 >= p and j0 + Int32(E) <= p + length:
                cute.autovec_copy(sI[None, w], frag)
                cute.autovec_copy(frag, gI[None, (d - Int64(p)) // Int64(E) + Int64(w)])
            else:
                for e in cutlass.range_constexpr(E):
                    j = j0 + Int32(e)
                    if j >= p and j < p + length:
                        mIndices[d + Int64(j - p)] = s_idx[j]

    @cute.kernel
    def rows_kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        rows: Int32,
        n: Int32,
        group: Int32,
    ):
        VEC = const_expr(self.vec)
        E = const_expr(self.ivec)
        ITEMS = const_expr(self.items)
        NUM_WARPS = const_expr(self.threads // 32)
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        gX = self.chunked(mX, VEC)
        gV = self.chunked(mValues, VEC)

        smem = cutlass.utils.SmemAllocator()
        s_idx = smem.allocate_tensor(
            self.index_dtype, cute.make_layout((self.tile + E,)), byte_alignment=16
        )
        s_row = smem.allocate_tensor(
            Int32, cute.make_layout((self.max_rows + 1,)), byte_alignment=16
        )
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((NUM_WARPS + 1,)), byte_alignment=16
        )

        r0 = bid * group
        nr = cutlass.min(group, rows - r0)
        length = nr * n
        lo = Int64(r0) * Int64(n)
        c0 = lo // Int64(VEC)
        shift = Int32(lo - c0 * Int64(VEC))
        q0 = tidx * Int32(ITEMS)
        q1 = (shift + length + Int32(VEC - 1)) // Int32(VEC)
        chunks = cute.make_rmem_tensor(cute.make_layout((VEC, ITEMS)), Uint8)
        masks = cute.make_rmem_tensor(ITEMS, Int32)
        f = self.load(gX, c0, q0, q1, length, shift, chunks, masks)
        excl, total = block_exclusive_prefix(f, s_scan, lane, warp, NUM_WARPS)

        # Prefix at each row start of the tile.
        ol0 = q0 * Int32(VEC) - shift
        start = cutlass.max(ol0, Int32(0))
        r = (start + n - Int32(1)) // n
        nxt = r * n
        fe = excl
        for v in cutlass.range_constexpr(ITEMS):
            for k in cutlass.range_constexpr(VEC):
                ol = ol0 + Int32(v * VEC + k)
                if ol == nxt and ol < length:
                    s_row[r] = fe
                    r = r + Int32(1)
                    nxt = nxt + n
                fe = fe + ((masks[v] >> Int32(k)) & Int32(1))
        if tidx == Int32(0):
            s_row[nr] = total
        cute.arch.barrier()

        p = Int32(lo % Int64(E))
        out = cute.make_rmem_tensor(VEC, Uint8)
        if start < length:
            r = start // n
            rstart = r * n
            pr = s_row[r]
            pr1 = s_row[r + Int32(1)]
            fe = excl
            for v in cutlass.range_constexpr(ITEMS):
                for k in cutlass.range_constexpr(VEC):
                    ol = ol0 + Int32(v * VEC + k)
                    out[k] = Uint8(0)
                    if ol >= Int32(0) and ol < length:
                        if ol == rstart + n:
                            r = r + Int32(1)
                            rstart = rstart + n
                            pr = pr1
                            pr1 = s_row[r + Int32(1)]
                        bit = (masks[v] >> Int32(k)) & Int32(1)
                        count = pr1 - pr
                        before = fe - pr
                        pos = before
                        if bit == Int32(0):
                            pos = count + ol - rstart - before
                        s_idx[p + rstart + pos] = self.index_dtype(ol - rstart)
                        if (ol - rstart < count) == self.descending:
                            out[k] = Uint8(1)
                        fe = fe + bit
                q = q0 + Int32(v)
                if q < q1:
                    base = q * Int32(VEC) - shift
                    self.store_values(mValues, gV, c0, q, base, Int32(0), length, out)
        cute.arch.barrier()
        self.write_runs(mIndices, s_idx, p, lo, length, p, lo, Int32(0))

    @cute.jit
    def row_bounds(self, n: Int32, group: Int32, segs: Int32, bid: Int32):
        VEC = const_expr(self.vec)
        row = bid // segs
        s = bid - row * segs
        rs = Int64(row) * Int64(n)
        c0 = rs // Int64(VEC)
        shift = Int32(rs - c0 * Int64(VEC))
        row_chunks = (shift + n + Int32(VEC - 1)) // Int32(VEC)
        cs = s * group
        ce = cutlass.min(cs + group, row_chunks)
        return row, s, rs, c0, shift, cs, ce

    @cute.jit
    def count(
        self,
        gX: cute.Tensor,
        c0: Int64,
        cs: Int32,
        ce: Int32,
        n: Int32,
        shift: Int32,
        s_scan: cute.Tensor,
    ) -> Int32:
        VEC = const_expr(self.vec)
        ITEMS = const_expr(self.items)
        tidx, _, _ = cute.arch.thread_idx()
        chunks = cute.make_rmem_tensor(cute.make_layout((VEC, ITEMS)), Uint8)
        masks = cute.make_rmem_tensor(ITEMS, Int32)
        f = Int32(0)
        for tile in cutlass.range(cs, ce, self.threads * ITEMS):
            q0 = tile + tidx * Int32(ITEMS)
            f = f + self.load(gX, c0, q0, ce, n, shift, chunks, masks)
        _, total = block_exclusive_prefix(
            f, s_scan, tidx % Int32(32), tidx // Int32(32), self.threads // 32
        )
        return total

    @cute.kernel
    def count_kernel(
        self, mX: cute.Tensor, mCounts: cute.Tensor, n: Int32, group: Int32, segs: Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((self.threads // 32 + 1,)), byte_alignment=16
        )
        # Last segments first: the scatter kernel then starts on the segments
        # still in L2.
        seg = nblocks - Int32(1) - bid
        row, _, _, c0, shift, cs, ce = self.row_bounds(n, group, segs, seg)
        total = self.count(self.chunked(mX, self.vec), c0, cs, ce, n, shift, s_scan)
        if tidx == Int32(0):
            cute.arch.atomic_add(mCounts.iterator + Int32(2) * nblocks + row, total)
            mCounts[seg] = total

    @cute.jit
    def lookback(self, mCounts: cute.Tensor, bid: Int32, s: Int32, s_scan: cute.Tensor):
        """First-class count of the row before segment ``s``.

        ``mCounts`` holds the count kernel's aggregate per segment, then an
        inclusive prefix plus one per segment (0: not yet published), then the
        row totals. Every aggregate is ready, so the look-back never waits."""
        tidx, _, _ = cute.arch.thread_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        lane = tidx % Int32(32)
        aggregates = bid - s
        prefixes = nblocks + aggregates
        if tidx < Int32(32):
            excl = Int32(0)
            j = s
            while j > Int32(0):
                k = j - Int32(1) - lane
                a = Int32(0)
                p = Int32(0)
                if k >= Int32(0):
                    a = mCounts[aggregates + k]
                    p = cute.arch.load(
                        mCounts.iterator + prefixes + k,
                        Int32,
                        sem="relaxed",
                        scope="gpu",
                    )
                hit = Int32(cute.arch.vote_ballot_sync(p != Int32(0)))
                if hit != Int32(0):
                    first = cute.arch.popc((hit & -hit) - Int32(1))
                    if lane > first:
                        a = Int32(0)
                    elif lane == first:
                        a = p - Int32(1)
                    j = Int32(0)
                excl = excl + cute.arch.shuffle_sync(warp_inclusive_prefix(a, lane), 31)
                j = j - Int32(32)
            if lane == Int32(0):
                own = mCounts[bid]
                cute.arch.store(
                    mCounts.iterator + prefixes + s,
                    excl + own + Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                s_scan[0] = excl
        cute.arch.barrier()
        excl = s_scan[0]
        cute.arch.barrier()
        return excl

    @cute.kernel
    def row_kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        mCounts: cute.Tensor,
        n: Int32,
        group: Int32,
        segs: Int32,
    ):
        VEC = const_expr(self.vec)
        E = const_expr(self.ivec)
        ITEMS = const_expr(self.items)
        NUM_THREADS = const_expr(self.threads)
        NUM_WARPS = const_expr(self.threads // 32)
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        gX = self.chunked(mX, VEC)
        gV = self.chunked(mValues, VEC)

        smem = cutlass.utils.SmemAllocator()
        s_idx = smem.allocate_tensor(
            self.index_dtype, cute.make_layout((self.tile + 2 * E,)), byte_alignment=16
        )
        s_scan = smem.allocate_tensor(
            Int32, cute.make_layout((NUM_WARPS + 1,)), byte_alignment=16
        )

        row, s, rs, c0, shift, cs, ce = self.row_bounds(n, group, segs, bid)
        row_f = Int32(0)
        first_run = Int32(0)
        if const_expr(self.mode == "split"):
            nblocks, _, _ = cute.arch.grid_dim()
            row_f = mCounts[Int32(2) * nblocks + row]
            first_run = self.lookback(mCounts, bid, s, s_scan)
        else:
            row_f = self.count(gX, c0, cs, ce, n, shift, s_scan)

        chunks = cute.make_rmem_tensor(cute.make_layout((VEC, ITEMS)), Uint8)
        out = cute.make_rmem_tensor(VEC, Uint8)
        masks = cute.make_rmem_tensor(ITEMS, Int32)
        for tile in cutlass.range(cs, ce, NUM_THREADS * ITEMS):
            q0 = tile + tidx * Int32(ITEMS)
            f = self.load(gX, c0, q0, ce, n, shift, chunks, masks)
            excl, tile_f = block_exclusive_prefix(f, s_scan, lane, warp, NUM_WARPS)
            tile_lo = cutlass.max(tile * Int32(VEC) - shift, Int32(0))
            tile_hi = cutlass.min(
                cutlass.min(tile + Int32(NUM_THREADS * ITEMS), ce) * Int32(VEC) - shift,
                n,
            )
            d1 = rs + Int64(first_run)
            d2 = rs + Int64(row_f + tile_lo - first_run)
            p1 = Int32(d1 % Int64(E))
            p2 = (p1 + tile_f + Int32(E - 1)) // Int32(E) * Int32(E)
            p2 = p2 + Int32(d2 % Int64(E))
            ol0 = q0 * Int32(VEC) - shift
            fe = excl
            for v in cutlass.range_constexpr(ITEMS):
                for k in cutlass.range_constexpr(VEC):
                    ol = ol0 + Int32(v * VEC + k)
                    out[k] = Uint8(0)
                    if ol >= tile_lo and ol < tile_hi:
                        bit = (masks[v] >> Int32(k)) & Int32(1)
                        slot = p1 + fe
                        if bit == Int32(0):
                            slot = p2 + ol - tile_lo - fe
                        s_idx[slot] = self.index_dtype(ol)
                        if (ol < row_f) == self.descending:
                            out[k] = Uint8(1)
                        fe = fe + bit
                q = q0 + Int32(v)
                if q < ce:
                    base = q * Int32(VEC) - shift
                    self.store_values(mValues, gV, c0, q, base, tile_lo, tile_hi, out)
            cute.arch.barrier()
            len2 = tile_hi - tile_lo - tile_f
            self.write_runs(mIndices, s_idx, p1, d1, tile_f, p2, d2, len2)
            cute.arch.barrier()
            first_run = first_run + tile_f


class Policy(NamedTuple):
    rows_threads: int
    rows_items: int  # most items per thread of a ``rows`` tile
    stage_bytes: int  # staged indices of a ``rows`` tile, at least one row
    max_rows: int  # rows per ``rows`` tile
    row_threads: int
    row_items: tuple[int, ...]  # ``row`` tiles, smallest whose rows fit L2
    split_threads: int
    split_items: int  # a ``split`` segment is one tile


# sm_120 is measured (see the PR notes); other arches scale the staging budget
# with shared memory per SM.
_POLICY = {120: Policy(256, 4, 32768, 1024, 512, (2, 4), 256, 2)}


def policy(arch: int, device: torch.device | None = None) -> Policy:
    if arch in _POLICY:
        return _POLICY[arch]
    return _POLICY[120]._replace(stage_bytes=caps(device).smem_per_sm // 3)


def spec_name(spec: dict) -> str:
    return (
        f"sort_partition_{spec['mode']}_{spec['index']}"
        f"_{'desc' if spec['descending'] else 'asc'}_t{spec['threads']}"
        f"_i{spec['items']}_v{spec['vec']}x{spec['ivec']}_r{spec['max_rows']}"
        f"_sm{spec['arch']}"
    )


def make_spec(
    arch: int,
    index_dtype: torch.dtype,
    descending: bool,
    mode: str,
    threads: int,
    items: int,
    vec: int,
    ivec: int,
    max_rows: int = 0,
) -> dict:
    return {
        "kernel": "partition",
        "mode": mode,
        "index": index_name(index_dtype),
        "descending": descending,
        "threads": threads,
        "items": items,
        "vec": vec,
        "ivec": ivec,
        "max_rows": max_rows,
        "arch": arch,
    }


def compile_args(spec: dict) -> tuple[PartitionSort, list]:
    """Kernel object and fake operands for one spec; shared by JIT and AOT."""
    index = INDEX_DTYPES[getattr(torch, spec["index"])]
    kernel = PartitionSort(
        index,
        spec["descending"],
        spec["mode"],
        spec["threads"],
        spec["items"],
        spec["vec"],
        spec["ivec"],
        spec["max_rows"],
    )
    vec, ivec = spec["vec"], spec["ivec"]
    fake = [
        fake_compact(Uint8, (sym(),), align=vec),
        fake_compact(Uint8, (sym(),), align=vec),
        fake_compact(index, (sym(),), align=ivec * index.width // 8),
        fake_compact(Int32, (sym(),)),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return kernel, fake


@instrumented_cutedsl_cache("aten::sort", key_fn=lambda spec: spec_name(dict(spec)))
def _compile(spec: tuple):
    kernel, fake = compile_args(dict(spec))
    return compile_kernel(kernel, *fake)


_FLAT_LIMIT = 2**31 - 64


def partition_sort(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    descending: bool,
    arch: int,
) -> None:
    """Stable sort of each row of 2D row-major bool ``x`` into ``values`` and
    ``indices``; also a valid unstable sort.

    The caller checks eligibility: contiguous outputs and rows that
    ``indices`` can address.
    """
    rows, n = x.shape
    if rows == 0 or n == 0:
        return
    # Flat extents are Int32.
    batch = max(1, _FLAT_LIMIT // n)
    if rows > batch:
        for start in range(0, rows, batch):
            rows_ = slice(start, start + batch)
            partition_sort(x[rows_], values[rows_], indices[rows_], descending, arch)
        return
    p = policy(arch, x.device)
    hw = caps(x.device)
    x_flat = x.view(torch.uint8).view(-1)
    values_flat = values.view(torch.uint8).view(-1)
    vec = min(supported_alignment(x_flat, 16), supported_alignment(values_flat, 16))
    ib = indices.element_size()
    ivec = supported_alignment(indices, 16) // ib
    chunks = -(-(n + vec - 1) // vec)
    counts = torch.empty(0, dtype=torch.int32, device=x.device)
    max_rows = 0
    if chunks <= p.rows_threads * p.rows_items:
        threads = p.rows_threads
        staged = p.stage_bytes // (threads * vec * ib)
        items = min(p.rows_items, max(-(-chunks // threads), staged))
        group = min(p.max_rows, (threads * items * vec - vec + 1) // n)
        mode, max_rows = "rows", p.max_rows
    else:
        mode, threads, items = "split", p.split_threads, p.split_items
        group = threads * items
        if rows >= hw.fill_blocks(p.row_threads, waves=2.0):
            # A CTA per row re-reads its row from L2 while the resident rows fit.
            for row_items in p.row_items:
                smem = p.row_threads * row_items * vec * ib
                per_sm = min(hw.blocks_per_sm(p.row_threads), hw.smem_per_sm // smem)
                if smem > hw.smem_per_block_optin:
                    break
                if hw.sm_count * per_sm * n <= hw.l2_bytes // 2:
                    mode, threads, items, group = (
                        "row",
                        p.row_threads,
                        row_items,
                        chunks,
                    )
                    break
        if mode == "split":
            segs = -(-chunks // group)
            counts = torch.zeros(
                rows * (2 * segs + 1), dtype=torch.int32, device=x.device
            )
    spec = make_spec(
        arch, indices.dtype, descending, mode, threads, items, vec, ivec, max_rows
    )
    _compile(tuple(sorted(spec.items())))(
        read_only(x_flat), values_flat, indices.view(-1), counts, rows, n, group
    )
