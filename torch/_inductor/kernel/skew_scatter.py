# mypy: allow-untyped-defs
"""
Skew-aware row scatter-add: ``out.index_add_(0, index, src)`` for 2-D ``out``.

``_skew_scatter_kernel`` is autotuned on each call site's first call between
``MODE=0``, the flat atomic scatter Inductor emits (kept unless another config is
at least 10% faster); ``MODE=1``, which samples the index for hot rows, sums them
in fp32 registers and adds each once per program; and ``MODE=2``, which sorts
each tile of indices and adds each distinct row once. The pick depends on the
index values, so it is never cached across processes. Small destinations instead
sum into private fp32 copies of ``out``. Without a first call on real data, sm80
takes ``_flat_scatter_kernel`` and sm90+ only small destinations.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch.library import wrap_triton


if TYPE_CHECKING:
    import triton
    import triton.language as tl
else:
    from torch._inductor.runtime.triton_compat import tl, triton


# Hot-row sampling: S samples, the first C of them candidates, up to K hot rows.
_HOT = {"S": 512, "C": 64, "K": 16, "MIN_COUNT": 6}


@triton.jit
def _seg_sum(k1, v1, k2, v2):
    return k2, v2 + tl.where(k1 == k2, v1, 0)


def _configs(flat: list[tuple[int, int, int, int]]) -> list[triton.Config]:
    """``flat`` (elements per program, warps, evict-first src, stages) configs first,
    as the autotuner's defaults; then the hot-row and tile kernels."""
    c = {"BLOCK": 512, "EVICT": 0, "RB": 16, "WAVES": 1, "TILE": 32}
    hot = [(16, 2, 8), (64, 4, 8)]
    cfgs = [
        triton.Config({**c, "MODE": 0, "BLOCK": b, "EVICT": e}, w, st)
        for b, w, e, st in flat
    ]
    cfgs += [
        triton.Config({**c, "MODE": 1, "RB": rb, "WAVES": v}, w) for rb, w, v in hot
    ]
    return cfgs + [triton.Config({**c, "MODE": 2}, 8)]


@triton.jit
def _flat_scatter_kernel(
    idx_ptr,
    src_ptr,
    out_ptr,
    N,
    R,
    stride_src,
    stride_out,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    INT64: tl.constexpr,
    EVEN: tl.constexpr,
    SRC_CONTIG: tl.constexpr,
    OUT_CONTIG: tl.constexpr,
    EVICT: tl.constexpr,
):
    # With EVEN, contiguous src and out and no EVICT, this is Inductor's own kernel.
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // D
    col = offs % D
    src = src_ptr + offs if SRC_CONTIG else src_ptr + row * stride_src + col
    if EVEN:
        m = None
        idx = tl.load(idx_ptr + row, eviction_policy="evict_last").to(tl.int64)
        if EVICT:
            v = tl.load(src, eviction_policy="evict_first")
        else:
            v = tl.load(src)
    else:
        m = row < N
        idx = tl.load(idx_ptr + row, mask=m, other=0, eviction_policy="evict_last")
        idx = idx.to(tl.int64)
        if EVICT:
            v = tl.load(src, mask=m, other=0.0, eviction_policy="evict_first")
        else:
            v = tl.load(src, mask=m, other=0.0)
    idx = tl.where(idx < 0, idx + R, idx)
    dst = out_ptr + idx * D + col if OUT_CONTIG else out_ptr + idx * stride_out + col
    tl.atomic_add(dst, v.to(out_ptr.dtype.element_ty), mask=m, sem="relaxed")


@functools.cache
def _get_skew_scatter_kernel(sm80: bool):
    # Built on first use: Triton-less builds have only a ``triton.jit`` stub.
    # Flat configs: Inductor's on sm90+; on sm80 one element per thread, where fp32
    # atomics issued several per thread run at about half the rate.
    flat = [(256, 8, 1, 3)] if sm80 else [(1024, 4, 0, 1), (512, 8, 0, 1)]

    # ``site`` (unused, a multiple of 16) keys the tuning per call site while every
    # site shares one compiled binary per config.
    @triton.autotune(
        configs=_configs(flat), key=["N", "D", "site"], restore_value=["out_ptr"]
    )
    @triton.jit
    def _skew_scatter_kernel(
        idx_ptr,
        src_ptr,
        out_ptr,
        N,
        R,
        stride_src,
        stride_out,
        sample_stride,
        site,
        D: tl.constexpr,
        DB: tl.constexpr,
        TDB: tl.constexpr,
        INT64: tl.constexpr,
        KEY64: tl.constexpr,
        EVEN: tl.constexpr,
        SRC_CONTIG: tl.constexpr,
        OUT_CONTIG: tl.constexpr,
        EVICT: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK: tl.constexpr,
        RB: tl.constexpr,
        WAVES: tl.constexpr,
        TILE: tl.constexpr,
        S: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        MIN_COUNT: tl.constexpr,
    ):
        if MODE == 0:
            _flat_scatter_kernel(
                idx_ptr,
                src_ptr,
                out_ptr,
                N,
                R,
                stride_src,
                stride_out,
                D,
                BLOCK,
                INT64,
                EVEN,
                SRC_CONTIG,
                OUT_CONTIG,
                EVICT,
            )
        elif MODE == 1:
            pid = tl.program_id(0)
            nprog = tl.num_programs(0)
            cols = tl.program_id(1) * DB + tl.arange(0, DB)
            cm = cols < D
            # Hot rows: the first C of S strided samples that occur in the samples
            # at least max(MIN_COUNT, 8x the uniform expectation) times.
            s = tl.arange(0, S).to(tl.int64) * sample_stride
            smp = tl.load(idx_ptr + s, mask=s < N, other=0).to(tl.int64)
            smp = tl.where(s < N, tl.where(smp < 0, smp + R, smp), -1)
            c = tl.arange(0, C)
            cp = c.to(tl.int64) * sample_stride
            cand = tl.load(idx_ptr + cp, mask=cp < N, other=0).to(tl.int64)
            cand = tl.where(cp < N, tl.where(cand < 0, cand + R, cand), -2)
            cnt = tl.sum((cand[:, None] == smp[None, :]).to(tl.int32), 1)
            dup = (cand[:, None] == cand[None, :]) & (c[None, :] < c[:, None])
            earlier = tl.sum(dup.to(tl.int32), 1)
            thresh = tl.maximum(MIN_COUNT, 8 * tl.minimum(N, S) // tl.maximum(R, 1))
            sel = (earlier == 0) & (cnt >= thresh) & (cand >= 0)
            slot = tl.cumsum(sel.to(tl.int32), 0) - 1
            kk = tl.arange(0, K)
            hit = (slot[None, :] == kk[:, None]) & sel[None, :]
            hot = tl.sum(tl.where(hit, cand[None, :] + 1, 0), 1) - 1
            n_hot = tl.sum((hot >= 0).to(tl.int32), 0)
            acc = tl.zeros([K, DB], dtype=tl.float32)
            for t in tl.range(pid, tl.cdiv(N, RB), nprog):
                rows = t.to(tl.int64) * RB + tl.arange(0, RB)
                m = rows < N
                idx = tl.load(idx_ptr + rows, mask=m, other=0).to(tl.int64)
                idx = tl.where(idx < 0, idx + R, idx)
                src = src_ptr + rows[:, None] * stride_src + cols[None, :]
                v = tl.load(src, mask=m[:, None] & cm[None, :], other=0.0)
                v = v.to(tl.float32)
                cold = m
                if n_hot > 0:
                    is_hot = tl.sum((idx[:, None] == hot[None, :]).to(tl.int32), 1)
                    cold = m & (is_hot == 0)
                    for k in range(n_hot):
                        hk = tl.sum(tl.where(kk == k, hot, 0), 0)
                        part = tl.sum(tl.where(((idx == hk) & m)[:, None], v, 0.0), 0)
                        acc += tl.where(kk[:, None] == k, part[None, :], 0.0)
                dst = out_ptr + idx[:, None] * stride_out + cols[None, :]
                v = v.to(out_ptr.dtype.element_ty)
                tl.atomic_add(dst, v, mask=cold[:, None] & cm[None, :], sem="relaxed")
            if n_hot > 0:
                dst = out_ptr + hot[:, None] * stride_out + cols[None, :]
                hm = (hot >= 0)[:, None] & cm[None, :]
                hacc = acc.to(out_ptr.dtype.element_ty)
                tl.atomic_add(dst, hacc, mask=hm, sem="relaxed")
        else:
            tile = tl.program_id(0)
            pos = tl.arange(0, TILE)
            rows = tile.to(tl.int64) * TILE + pos
            m = rows < N
            idx = tl.load(idx_ptr + rows, mask=m, other=0).to(tl.int64)
            idx = tl.where(idx < 0, idx + R, idx)
            # (row, position) pairs sorted by row; positions past N sort last (row R)
            key = tl.where(m, idx, R) * TILE + pos
            if not KEY64:
                key = key.to(tl.int32)
            s = tl.sort(key)
            srow = (s // TILE).to(tl.int64)
            valid = srow < R
            # occurrences of each position's row at this or later positions: 1 = last
            ones = tl.full([TILE], 1, tl.int32)
            _keys, left = tl.associative_scan((tl.flip(srow, 0), ones), 0, _seg_sum)
            tail = valid & (tl.flip(left, 0) == 1)
            repeats = tl.sum(tail.to(tl.int32), 0) < tl.sum(m.to(tl.int32), 0)
            spos = tile.to(tl.int64) * TILE + (s % TILE).to(tl.int64)
            for c0 in range(0, D, TDB):
                cols = c0 + tl.arange(0, TDB)
                cm = cols < D
                if repeats:
                    src = src_ptr + spos[:, None] * stride_src + cols[None, :]
                    vs = tl.load(src, mask=valid[:, None] & cm[None, :], other=0.0)
                    vs = vs.to(tl.float32)
                    keys = tl.broadcast_to(srow[:, None], (TILE, TDB))
                    _rows, acc = tl.associative_scan((keys, vs), 0, _seg_sum)
                    dst = out_ptr + srow[:, None] * stride_out + cols[None, :]
                    acc = acc.to(out_ptr.dtype.element_ty)
                    tm = tail[:, None] & cm[None, :]
                    tl.atomic_add(dst, acc, mask=tm, sem="relaxed")
                else:
                    src = src_ptr + rows[:, None] * stride_src + cols[None, :]
                    v = tl.load(src, mask=m[:, None] & cm[None, :], other=0.0)
                    dst = out_ptr + idx[:, None] * stride_out + cols[None, :]
                    v = v.to(out_ptr.dtype.element_ty)
                    tl.atomic_add(dst, v, mask=m[:, None] & cm[None, :], sem="relaxed")

    # Read by Inductor: never cache the pick, and prefer the flat configs.
    _skew_scatter_kernel.fn.autotune_per_process = True
    _skew_scatter_kernel.fn.autotune_default_configs = len(flat)
    _skew_scatter_kernel.fn.autotune_default_margin = 0.1
    return _skew_scatter_kernel


@triton.jit
def _parts_scatter_kernel(
    idx_ptr,
    src_ptr,
    out_ptr,
    N,
    R,
    stride_src,
    part_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // D
    col = offs % D
    m = row < N
    idx = tl.load(idx_ptr + row, mask=m, other=0)
    idx = tl.where(idx < 0, idx + R, idx)
    m = m & (idx >= 0) & (idx < R)
    v = tl.load(src_ptr + row * stride_src + col, mask=m, other=0.0).to(tl.float32)
    po = out_ptr + (pid % PARTS) * part_stride + idx * D + col
    tl.atomic_add(po, v, mask=m, sem="relaxed")


def _num_parts(n: int, r: int, d: int) -> int:
    """Private copies of ``out`` for small destinations with many adds per row."""
    if r * d > (1 << 12) or n < 16 * r:
        return 1
    p = (1 << 21) // max(1, r * d)
    return 1 << (min(4096, p).bit_length() - 1)


def _parts_scatter(
    index: torch.Tensor, src: torch.Tensor, r: int, parts: int
) -> torch.Tensor:
    n, d = index.numel(), src.shape[1]
    target = torch.zeros((parts, r, d), dtype=torch.float32, device=src.device)
    block = 512
    int64 = max(n * max(src.stride(0), d), parts * r * d) >= 2**31 - block
    meta = {"D": d, "BLOCK": block, "PARTS": parts, "INT64": int64, "num_warps": 4}
    grid = (triton.cdiv(n * d, block),)
    wrap_triton(_parts_scatter_kernel)[grid](
        index, src, target, n, r, src.stride(0), r * d, **meta
    )
    return target.sum(0)


def _int64(n: int, r: int, d: int, src: torch.Tensor, out: torch.Tensor) -> bool:
    return max(n * max(src.stride(0), d), r * out.stride(0)) >= 2**31 - 4096


def _flat_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    n = index.numel()
    r, d = out.shape
    block = 256
    strides = (src.stride(0), out.stride(0))
    meta = dict(EVEN=False, SRC_CONTIG=False, OUT_CONTIG=False, EVICT=True, num_warps=8)
    i64 = _int64(n, r, d, src, out)
    wrap_triton(_flat_scatter_kernel)[(triton.cdiv(n * d, block),)](
        index, src, out, n, r, *strides, D=d, BLOCK=block, INT64=i64, **meta
    )


def _launch(launcher, out, index, src, site: int, **meta) -> None:
    n = index.numel()
    r, d = out.shape
    tdb = min(triton.next_power_of_2(d), 8)
    strides = (src.stride(0), out.stride(0))
    extra = (max(1, n // _HOT["S"]), 16 * (site + 1))
    launcher(index, src, out, n, r, *strides, *extra, D=d, TDB=tdb, **_HOT, **meta)


def _skew_scatter(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int
) -> None:
    n = index.numel()
    r, d = out.shape
    db = max(16, min(triton.next_power_of_2(d), 128))
    nd = triton.cdiv(d, db)
    sms = torch.cuda.get_device_properties(out.device).multi_processor_count

    def grid(meta):
        if meta["MODE"] == 0:
            return (triton.cdiv(n * d, meta["BLOCK"]), 1)
        if meta["MODE"] == 1:
            programs = min(triton.cdiv(n, meta["RB"]), sms * meta["WAVES"] // nd)
            return (max(1, programs), nd)
        return (triton.cdiv(n, meta["TILE"]), 1)

    sm80 = torch.cuda.get_device_capability(out.device)[0] == 8
    kernel = _get_skew_scatter_kernel(sm80)
    meta = dict(DB=db, INT64=_int64(n, r, d, src, out), KEY64=(r + 1) * 64 >= 2**31)
    meta["EVEN"] = not sm80 and (n * d) % 1024 == 0
    meta["SRC_CONTIG"] = not sm80 and src.stride(0) == d
    meta["OUT_CONTIG"] = not sm80 and out.stride(0) == d
    _launch(wrap_triton(kernel)[grid], out, index, src, site, **meta)


def _tile_scatter(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    """The ``MODE=2`` path alone (tests)."""
    kernel = _get_skew_scatter_kernel(False)
    (cfg,) = [c for c in kernel.configs if c.kwargs["MODE"] == 2]
    meta = dict(cfg.kwargs, DB=16, KEY64=(out.shape[0] + 1) * 32 >= 2**31)
    meta.update(dict.fromkeys(("INT64", "EVEN", "SRC_CONTIG", "OUT_CONTIG"), False))
    grid = (triton.cdiv(index.numel(), 32), 1)
    _launch(kernel.fn[grid], out, index, src, 0, **meta, num_warps=8)


def rewrite_supported(
    device: torch.device, dtype: torch.dtype, rows: int, d: int, n: int, autotune: bool
) -> bool:
    """Small destinations: any CUDA arch (fp32 outputs only on sm80). Otherwise
    fp32 outputs only: with ``autotune`` on sm80+, without it on sm80 only."""
    cc = torch.cuda.get_device_capability(device)
    if _num_parts(n, rows, d) > 1:
        return dtype == torch.float32 or cc >= (9, 0)
    if dtype != torch.float32 or cc < (8, 0):
        return False
    return autotune or cc[0] == 8


def _row_scatter(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int
) -> None:
    """``site`` < 0: no first-call autotuning (the flat kernel)."""
    if site >= 0:
        _skew_scatter(out, index, src, site)
    else:
        _flat_scatter(out, index, src)


def _supported(out_dtype: torch.dtype, index: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        src.is_cuda
        and torch.version.hip is None
        and src.dim() == 2
        and src.stride(1) == 1
        and index.numel() > 0
        and src.is_floating_point()
        and out_dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


def _accumulate_fp32(out_dtype: torch.dtype, rows: int, src: torch.Tensor) -> bool:
    # 16-bit outputs small next to src accumulate in fp32, rounded once.
    return out_dtype != torch.float32 and 2 * rows * src.shape[1] <= src.numel()


def scatter_rows(
    index: torch.Tensor,
    src: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
    site: int = -1,
) -> torch.Tensor:
    """``zeros(rows, D, dtype=dtype).index_add_(0, index, src)``."""
    d = src.shape[-1]
    if not _supported(dtype, index, src):
        out = torch.zeros((rows, d), dtype=dtype, device=src.device)
        return out.index_add_(0, index.reshape(-1), src.to(dtype))
    index = index.reshape(-1)
    parts = _num_parts(index.numel(), rows, d)
    if parts > 1:
        return _parts_scatter(index, src, rows, parts).to(dtype)
    acc_fp32 = _accumulate_fp32(dtype, rows, src)
    acc_dtype = torch.float32 if acc_fp32 else dtype
    out = torch.zeros((rows, d), dtype=acc_dtype, device=src.device)
    _row_scatter(out, index, src, site)
    return out.to(dtype) if acc_fp32 else out


def skew_scatter_add_(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int = -1
) -> torch.Tensor:
    """In-place ``out.index_add_(0, index, src)``; ``site`` >= 0 names the call site
    for first-call autotuning."""
    if not _supported(out.dtype, index, src) or out.dim() != 2 or out.stride(1) != 1:
        return out.index_add_(0, index, src.to(out.dtype))
    index = index.reshape(-1)
    r, d = out.shape
    parts = _num_parts(index.numel(), r, d)
    if parts > 1:
        return out.add_(_parts_scatter(index, src, r, parts).to(out.dtype))
    if _accumulate_fp32(out.dtype, r, src):
        acc = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
        _row_scatter(acc, index, src, site)
        return out.add_(acc)
    _row_scatter(out, index, src, site)
    return out
