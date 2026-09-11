# Row-reduction launch policy and plan cache for tile.TileReduce. Runtime loops share
# each kernel across a vector class; narrow rows may use one thread and TMA staging.
import math
from typing import NamedTuple

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
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# Legal power-of-two reduction/block widths, widest last.
_TPR_RUNGS = tuple(t for _, t in _TPR_LADDER) + (_TPR_MAX,)
# Small rows use fewer threads/block.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Rows >=16 KB need 256 threads; the element ladder underthreads them by 1.1-1.4x.
_WIDE_ROW_BYTES = 16 * 1024


# Narrow rows: merged mappings floor tpr at one warp, wasting lanes and measuring 4.0x
# slower at (1048576, 32). tpr=1 serves any trait without merging; MAX_UNROLL bounds
# its whole-row unroll above the measured crossover.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# Measured (minimum rows, vector-chunk budget) ladder: tpr=1 shrinks the grid and
# needs more rows as rows widen. Tiers generalize across dtypes and measured 1.14-1.87x
# at M=4096, up to 33.7x at M=262144.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# Direct tpr=1 loads over-fetch once adjacent rows no longer share a 128-byte line:
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


def tma_ok(N: int, itemsize: int, M: int, device=None) -> bool:
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
    # Row-kernel defaults; explicit arguments override them.
    tpr: int  # threads per row
    nt: int  # threads per block


def row_config(N: int, dtype_width: int) -> "_RowConfig":
    # Occupancy by N and dtype; no nfields rule fits opposing fp32/bf16 optima.
    # Check the byte rung first because the element ladder underthreads it by ~1.3x.
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


def single_row_config(N: int, dtype_width: int):
    # A lone row cannot exploit packing, so use its widest feedable legal rung. Computed
    # widths changed tree shape and returned wrong variance. Measured var_mean improves
    # from 0.53-0.93x to 1.47-1.62x of ATen.
    cfg = row_config(N, dtype_width)
    vec = math.gcd(N, 128 // dtype_width)
    feedable = min(_TPR_MAX, N // max(1, vec))  # vector loads this row can issue
    rungs = [t for t in _TPR_RUNGS if WARP <= t <= feedable]
    if not rungs or rungs[-1] <= cfg.tpr:
        return None
    return _RowConfig(tpr=rungs[-1], nt=rungs[-1])


def _declared_align(x, natural: int) -> int:
    """Return the greatest N-allowed alignment met by `x`'s base pointer."""
    # const_data_ptr, so reading the address does not materialize a COW tensor.
    with torch._C.DisableTorchFunctionSubclass():
        ptr = x.const_data_ptr()
    align = natural
    while align > x.element_size() and ptr % align:
        align //= 2
    return align


def reduce_row_tile(
    trait,
    trait_key,
    x,
    out_dtypes,
    nouts=1,
    tpr=None,
    nt=None,
    final=True,
    unroll=None,
    use_tma=None,
):
    """Reduce 2-D `x` rows, returning outputs or raw field partials when `final=False`."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    cfg = row_config(N, x.element_size() * 8)
    # Scalar rows use 16 to hide narrow-load latency; vectorized rows use 4, at or near
    # the measured optimum across 4/8/16/32.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    tpr = max(WARP, cfg.tpr) if tpr is None else tpr
    nt = max(tpr, cfg.nt) if nt is None else nt
    nt -= nt % tpr  # rows_per_block must be whole
    if use_tma is None:
        natural = tile.align_bytes(N, x.element_size())
        use_tma = (
            tpr == 1
            and _declared_align(x, natural) == natural
            and tma_ok(N, x.element_size(), M, x.device)
        )
    dt = torch2cute[x.dtype]
    op = tile.TileReduce(
        trait,
        dt,
        "row",
        N,
        tpr=tpr,
        nt=nt,
        nouts=nouts,
        final=final,
        unroll=unroll,
        use_tma=use_tma,
    )

    # Final projects nouts; stage 1 stores one raw buffer per field.
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))
    # Declare alignment to retain wide loads (worth 3x), narrowed for storage offsets
    # outside TMA. Runtime folds share a vector class; TMA bakes its box width.
    isz = x.element_size()
    align = (
        op.tilemap.align_bytes(isz)
        if use_tma
        else _declared_align(x, tile.align_bytes(N, isz))
    )

    def _fake():
        # TMA bakes N; runtime folds share a vector class. None omits unused column args.
        inner = N if use_tma else _L.sym(op.vec)
        return (
            [_L.fake_compact(dt, (_L.sym(), inner), stride_order=(1, 0), align=align)],
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
