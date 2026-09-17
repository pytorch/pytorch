# Row-reduction launch policy and plan cache for tile.TileReduce. Runtime loops share
# each kernel across a vector class; narrow rows may use one thread and TMA staging.
import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import cutlass.cute as cute
from cutlass import Int32

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import tile
from .traits import WARP


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}


# First matching B200 threads-per-row anchor wins; small rows pack without cross-warp merge.
_THREADS_PER_ROW_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_MAX_THREADS_PER_ROW = 256
# Legal power-of-two reduction/block widths, widest last.
_THREADS_PER_ROW_RUNGS = tuple(t for _, t in _THREADS_PER_ROW_LADDER) + (
    _MAX_THREADS_PER_ROW,
)
# Small rows use fewer threads/block.
_SMALL_THREADS_PER_BLOCK, _LARGE_THREADS_PER_BLOCK = 128, 256
_THREADS_PER_BLOCK_GATE_N = 16 * 1024
# Rows >=16 KB need 256 threads; the element ladder underthreads them by 1.1-1.4x.
_WIDE_ROW_BYTES = 16 * 1024


# Narrow rows: merged mappings floor threads_per_row at one warp, wasting lanes and measuring 4.0x
# slower at (1048576, 32). threads_per_row=1 serves any trait without merging; MAX_UNROLL bounds
# its whole-row unroll above the measured crossover.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# Measured (minimum rows, vector-chunk budget) ladder: threads_per_row=1 shrinks the grid and
# needs more rows as rows widen. Tiers generalize across dtypes and measured 1.14-1.87x
# at M=4096, up to 33.7x at M=262144.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# Direct threads_per_row=1 loads over-fetch once adjacent rows no longer share a 128-byte line:
# 7001 GB/s at N=16 versus 4584 at N=32. TMA with smem rotation gains 1.49-1.86x;
# the rotation mask requires power-of-two fp32 N.
_TMA_MIN_STRIDE = 128


def narrow_row(N: int, itemsize: int, M: int) -> bool:
    """Is this geometry in the regime where one thread per row beats the packed shape?"""
    if N < 1 or N > _MAX_NARROW_N:
        return False
    chunks = N // tile.vec_size(N, itemsize)
    for min_rows, budget in _CHUNK_LADDER:
        if M >= min_rows:
            return chunks <= budget
    return False


def tma_ok(
    N: int,
    itemsize: int,
    M: int,
    device: torch.device | int | str | None = None,
) -> bool:
    """Should this geometry stage its load through TMA rather than load direct?"""
    if itemsize != 4 or N <= 0 or N & (N - 1) or N * itemsize < _TMA_MIN_STRIDE:
        return False
    if not narrow_row(N, itemsize, M):
        return False
    if device is not None:
        # This runs before every plan lookup; memoize the ~1.3us device query.
        from ...cutedsl import hw_caps as _hw

        if _hw.caps(device).cc[0] < 9:
            return False  # TMA is sm_90+
    return True


class _RowConfig(NamedTuple):
    """Row-kernel defaults; explicit arguments override them."""

    threads_per_row: int  # threads per row
    threads_per_block: int  # threads per block


def row_config(N: int, dtype_width: int) -> _RowConfig:
    """Choose occupancy by N and dtype.

    The byte rung takes priority because the element ladder underthreads it by about
    1.3x. No field-count rule fits the opposing fp32 and bf16 optima.
    """
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES:
        return _RowConfig(
            threads_per_row=_MAX_THREADS_PER_ROW,
            threads_per_block=_LARGE_THREADS_PER_BLOCK,
        )
    threads_per_row = next(
        (threads for limit, threads in _THREADS_PER_ROW_LADDER if N <= limit),
        _MAX_THREADS_PER_ROW,
    )
    threads_per_block = (
        _SMALL_THREADS_PER_BLOCK
        if N <= _THREADS_PER_BLOCK_GATE_N
        else _LARGE_THREADS_PER_BLOCK
    )
    return _RowConfig(
        threads_per_row=threads_per_row, threads_per_block=threads_per_block
    )


def single_row_config(N: int, dtype_width: int) -> _RowConfig | None:
    """Choose the widest feedable legal rung for a single-row launch.

    Computed widths changed the tree and returned wrong variance. This improves
    measured var_mean from 0.53-0.93x to 1.47-1.62x of ATen.
    """
    cfg = row_config(N, dtype_width)
    vec = math.gcd(N, 128 // dtype_width)
    feedable = min(
        _MAX_THREADS_PER_ROW, N // max(1, vec)
    )  # vector loads this row can issue
    rungs = [
        threads for threads in _THREADS_PER_ROW_RUNGS if WARP <= threads <= feedable
    ]
    if not rungs or rungs[-1] <= cfg.threads_per_row:
        return None
    return _RowConfig(threads_per_row=rungs[-1], threads_per_block=rungs[-1])


def reduce_row_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int = 1,
    threads_per_row: int | None = None,
    threads_per_block: int | None = None,
    final: bool = True,
    unroll: int | None = None,
    use_tma: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce 2-D `x` rows, returning outputs or raw field partials when `final=False`."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    cfg = row_config(N, x.element_size() * 8)
    # Scalar rows use 16 to hide narrow-load latency; vectorized rows use 4, at or near
    # the measured optimum across 4/8/16/32.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    threads_per_row = (
        max(WARP, cfg.threads_per_row) if threads_per_row is None else threads_per_row
    )
    threads_per_block = (
        max(threads_per_row, cfg.threads_per_block)
        if threads_per_block is None
        else threads_per_block
    )
    threads_per_block -= (
        threads_per_block % threads_per_row
    )  # rows_per_block must be whole
    isz = x.element_size()
    tma_base_aligned = (
        _L.supported_alignment(x, tile.TRANSFER_ALIGNMENT) == tile.TRANSFER_ALIGNMENT
    )
    tma_stride_bytes = x.stride(0) * isz
    tma_stride_aligned = tma_stride_bytes % tile.TRANSFER_ALIGNMENT == 0
    if use_tma is None:
        use_tma = (
            threads_per_row == 1
            and tma_base_aligned
            and tma_stride_aligned
            and tma_ok(N, isz, M, x.device)
        )
    elif use_tma and not tma_base_aligned:
        raise ValueError(f"TMA requires a {tile.TRANSFER_ALIGNMENT}-byte aligned input")
    elif use_tma and not tma_stride_aligned:
        raise ValueError(
            f"TMA requires a {tile.TRANSFER_ALIGNMENT}-byte aligned row stride, "
            f"got {tma_stride_bytes} bytes"
        )
    dt = torch2cute[x.dtype]
    op = tile.TileReduce(
        trait,
        dt,
        "row",
        N,
        threads_per_row=threads_per_row,
        threads_per_block=threads_per_block,
        nouts=nouts,
        final=final,
        unroll=unroll,
        use_tma=use_tma,
    )

    # Final projects nouts; stage 1 stores one raw buffer per field.
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / threads_per_row))
    # Declare alignment to retain wide loads (worth 3x), narrowed for storage offsets
    # outside TMA. Runtime folds share a vector class; TMA bakes its box width.
    align = (
        tile.TRANSFER_ALIGNMENT
        if use_tma
        else _L.supported_alignment(x, tile.align_bytes(N, isz))
    )

    def _fake():
        # TMA bakes N; runtime folds share a vector class. None omits unused column args.
        if use_tma:
            fake_in = cute.runtime.make_fake_tensor(
                dt,
                (_L.sym(), N),
                (
                    cute.sym_int64(divisibility=tile.TRANSFER_ALIGNMENT // isz),
                    1,
                ),
                assumed_align=align,
            )
        else:
            fake_in = _L.fake_compact(
                dt,
                (_L.sym(), _L.sym(op.vec)),
                stride_order=(1, 0),
                align=align,
            )
        return (
            [fake_in],
            [_L.fake_compact(torch2cute[o.dtype], (_L.sym(),)) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            None,
            None,
            _stream(),
        )

    # Pointer-dependent alignment is compiled, so include it in the key.
    dts = tuple(out_dtypes[:ndst])
    key = ("rowtile", trait_key, x.dtype, dts, align) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    # read_only avoids COW materialization; None omits unused column arguments.
    fn([_L.read_only(x)], list(outs), nchunks, nwaves, Int32(N), None, None, _stream())
    return tuple(outs)
