# Row-reduction launch policy and plan cache for tile.TileReduce. Runtime loops share
# each kernel across a vector class; narrow rows may use one thread and TMA staging.
import math
import os
from collections.abc import Sequence
from typing import Any, cast, Literal, NamedTuple

import cutlass.cute as cute
from cutlass import Int32

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import tile
from .traits import WARP, welford_nouts


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}
_INT32_LIMIT = 1 << 31


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
# H100 direct loads win below this; wider one-thread rows need TMA.
_HOPPER_DIRECT_ROW_BYTES = 384


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
        if _hw.caps(device).cc[0] < 9:
            return False  # TMA is sm_90+
    return True


def one_thread_row_ok(
    N: int,
    itemsize: int,
    M: int,
    device: torch.device | int | str,
) -> bool:
    if not narrow_row(N, itemsize, M):
        return False
    return (
        _hw.caps(device).cc != (9, 0)
        or N * itemsize < _HOPPER_DIRECT_ROW_BYTES
        or tma_ok(N, itemsize, M, device)
    )


# --- INNER-TREE ORDER (opt-in) --- Unlike launch-shape orders, this fixes the DAG from N,
# making it hash-pinnable, and covers every N to prevent silent fallback. It costs
# 0.92-1.41x the rolled fold and requires per-N compilation. The native gate is live for tests;
# the published sum gate also selects this order for geometries its adapter cannot serve.
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
_SUM_INNER_TREE_ENV = "PYTORCH_SUM_INNER_TREE"
# Block size for both one-thread-per-row shapes and fixed multirow carry cap.
_MULTIROW_ROWS_PER_BLOCK = 128
_MULTIROW_MAX_DEPTH = 6

# Bit-neutral tuning knobs keyed by architecture. They affect launch geometry and
# staging, and are part of `_ItreePlan.sig`.


class _ItreeArch(NamedTuple):
    split_stage_rows: int  # independent stage-1 batches staged per block
    combine_rpb: int  # rows per block for the stage-2 combine
    combine_group_bytes: int  # per iteration of its chain; 0 folds one link at a time
    stage_rows: bool  # stage the one-thread-per-row shape's block through smem
    # Most cp.async per lane and row; 0 disables combine staging.
    combine_async: int
    combine_max: int  # maximum partials staged per row
    combine_unroll: int = (
        0  # staged fold unroll factor; 0 retains full static unrolling
    )
    uniform_count: bool = False
    combine_weights: bool = False


# Performance tuning is keyed by full compute capability. Each measured GPU has
# an explicit row so changing the default cannot retune it.
_ITREE_ARCH: dict[str | tuple[int, int], _ItreeArch] = {
    "default": _ItreeArch(
        split_stage_rows=0,
        combine_rpb=128,
        combine_group_bytes=0,
        stage_rows=False,
        combine_async=0,
        combine_max=512,
    ),
    (9, 0): _ItreeArch(
        split_stage_rows=0,
        combine_rpb=WARP,
        combine_group_bytes=16,
        stage_rows=True,
        combine_async=8,
        combine_max=512,
    ),
    # B200 is the tuning anchor for the other constants in this file.
    # A 2K combine tile saves 1.1-2.5us at 128 MiB; 4K regresses 16-bit sums.
    (10, 0): _ItreeArch(
        split_stage_rows=4,
        combine_rpb=WARP,
        combine_group_bytes=16,
        stage_rows=False,
        combine_async=32,
        combine_max=2048,
    ),
    # Pin Rubin's measured baseline independently of future default changes.
    (10, 7): _ItreeArch(
        split_stage_rows=0,
        combine_rpb=128,
        combine_group_bytes=0,
        stage_rows=False,
        combine_async=0,
        combine_max=512,
    ),
}


def _itree_arch(
    device: torch.device | int | str | None = None,
) -> _ItreeArch:
    """Return tuning for `device`, defaulting safely because all knobs are bit-neutral."""
    return _ITREE_ARCH.get(_hw.caps(device).cc, _ITREE_ARCH["default"])


def select_full_itree_arch(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    N: int,
    M: int,
    *,
    field_bits: tuple[int, ...],
    out_dtypes: tuple[torch.dtype, ...],
    alignment: int,
    contiguous: bool,
) -> _ItreeArch | None:
    if (
        cc != (10, 7)
        or dtype not in (torch.float32, torch.bfloat16)
        or M != 1
        or not contiguous
        or alignment < tile.TRANSFER_ALIGNMENT
    ):
        return None
    itemsize = 4 if dtype == torch.float32 else 2
    if N * itemsize not in (256 << 20, 2 << 30):
        return None
    if trait_key == "sum" and field_bits == (32,) and out_dtypes == (torch.float32,):
        unroll = 0
    elif (
        trait_key == "argmaxi32"
        and field_bits == (32, 32)
        and out_dtypes == (torch.int64,)
    ):
        unroll = 128
    elif (
        field_bits == (32, 32, 32)
        and welford_nouts(trait_key)
        and out_dtypes == (dtype,) * welford_nouts(trait_key)
    ):
        unroll = 64
    else:
        return None
    # Measured full-reduction anchors; keep columns and other sizes on their arch profile.
    welford = field_bits == (32, 32, 32)
    return _ItreeArch(
        4,
        WARP,
        16,
        False,
        32,
        2048,
        unroll,
        uniform_count=welford,
        combine_weights=welford,
    )


def _full_itree_arch(
    trait: Any, trait_key: str, x: torch.Tensor, out_dtypes: Sequence[torch.dtype]
) -> _ItreeArch | None:
    M, N = x.shape
    if M != 1 or N * x.element_size() not in (256 << 20, 2 << 30):
        return None
    return select_full_itree_arch(
        _hw.caps(x.device).cc,
        x.dtype,
        trait_key,
        N,
        M,
        field_bits=tuple(dt.width for dt in trait.fdtypes),
        out_dtypes=tuple(out_dtypes),
        alignment=_L.supported_alignment(x, tile.TRANSFER_ALIGNMENT),
        contiguous=x.is_contiguous(),
    )


# Default per-thread width for the order, in elements (k*vec*eff live in registers at once).
# Fusing adjacent chunks keeps a full warp, and therefore coalesced loads, on each.
_ITREE_THREAD_ELEMS = 64

# Vector-width multiplier; each doubling costs ~2x. `bte`, not this, fixes the bits.
_ITREE_VEC_MUL = 1

# Occupancy-only block target; 64 threads starves small-N rows (145.3us versus 68.2us).
_ITREE_BLOCK_THREADS = 256

# Smem resolves coalesced vec loads versus contiguous per-lane trees, replacing butterflies
# without changing bits. cp.async tiles cut (65536, 1024) from 48.2 to 40.9us. Single-batch
# full runs only: shorter runs, register staging, and untiled buffers lost. 32 columns/lane
# reached 40.7us at N=1024 and caps wider rows at 4.6 KB smem each.
_ITREE_STAGE_E = 32


def inner_tree_order_enabled() -> bool:
    """Is the reproducible-DAG order requested? Read live, so tests can toggle it."""
    return any(
        os.environ.get(name, "") not in ("", "0")
        for name in (_INNER_TREE_ENV, _SUM_INNER_TREE_ENV)
    )


def reduction_order(
    order: Literal["unordered", "inner_tree"] | None = None,
) -> Literal["unordered", "inner_tree"]:
    if order is None:
        return "inner_tree" if inner_tree_order_enabled() else "unordered"
    if order not in ("unordered", "inner_tree"):
        raise ValueError(f"unknown reduction order: {order!r}")
    return order


def select_row_order(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    N: int,
    M: int,
    *,
    order: Literal["unordered", "inner_tree"],
    nfields: int,
    nouts: int,
    acc_bits: int = 32,
    alignment: int = 16,
) -> Literal["linear", "inner_tree"]:
    if order == "inner_tree":
        return "inner_tree"
    if order != "unordered":
        raise ValueError(f"unknown reduction order: {order!r}")
    if (
        cc != (10, 7)
        or dtype not in (torch.float32, torch.bfloat16)
        or N not in (256, 1024, 4096)
        or acc_bits != 32
        or alignment < 16
    ):
        return "linear"
    itemsize = 4 if dtype == torch.float32 else 2
    # Seed measured anchors; intermediate sizes retain their existing policy.
    size = M * N * itemsize
    if size not in (16 << 20, 64 << 20, 256 << 20, 2 << 30):
        return "linear"
    welford = nfields == 3 and welford_nouts(trait_key) == nouts
    argmax = (trait_key, nfields, nouts) == ("argmaxi32", 2, 1)
    simple = (
        nfields == 1 and nouts == 1 and trait_key in ("sum", "mean", "amax", "vnorm2")
    )
    if not (simple or argmax or welford):
        return "linear"
    if N == 1024:
        inner = welford or (dtype == torch.bfloat16 and argmax)
    elif N == 256:
        if dtype == torch.bfloat16:
            inner = size == 16 << 20 and not argmax
        else:
            inner = size >= 256 << 20 or (
                not argmax and (trait_key != "mean" or size >= 64 << 20)
            )
    else:
        inner = (
            welford
            or argmax
            or (dtype == torch.float32 and (trait_key == "mean" or size >= 256 << 20))
        )
    if inner:
        return "inner_tree"
    return "linear"


class _ItreePlan(NamedTuple):
    """Compile-time DAG state for one of upstream's three N-selected shapes.

    Shape is not a launch preference. Split's runtime batch index has two distinct widths.
    """

    shape: str
    vec: int
    wpr: int  # warps cooperating on one row; 0 == one thread per row
    rows_per_block: int
    depth: int
    batches: tuple[tuple[int, int, int, int], ...]
    tms: tuple[tile.TileMap, ...]
    # split only: (nbatch, batch_total_elements, last_remaining, chunk_full, chunk_last)
    split: tuple[int, ...] = ()
    # Adjacent chunks fused without changing their trees or cross-chunk merge.
    kchunk: int = 1
    # Fold each thread run linearly instead of as a tree, changing the DAG.
    vec_linear: bool = False
    # Staged columns/lane; 0 disables. One butterfly preserves ATen's chunk nesting.
    stage_e: int = 0
    # Stage the block's rows through smem where the global read is narrower than a full 32-lane
    # transfer. Bit-neutral: only where the bytes come from moves, not the tile map or any add.
    stage_rows: bool = False
    # COMBINE shape only: partials pulled into registers per iteration of its linear chain. 1 folds
    # link at a time. Bit-neutral -- the chain keeps its association either way.
    combine_grp: int = 1
    # COMBINE shape only: partials per row per cp.async-staged smem tile. 0 folds straight from
    # global. Also bit-neutral -- staging moves where the operand is read from, not the chain.
    combine_tile: int = 0
    combine_unroll: int = 0
    # Nonzero only for equal, nonempty, power-of-two Welford partial counts.
    combine_count: int = 0
    combine_weights: bool = False

    @property
    def sig(self) -> tuple[Any, ...]:
        return (
            self.shape,
            self.vec,
            self.wpr,
            self.rows_per_block,
            self.depth,
            self.batches,
            tuple(t.sig for t in self.tms),
            self.split,
            self.kchunk,
            self.vec_linear,
            self.stage_e,
            self.stage_rows,
            self.combine_grp,
            self.combine_tile,
            self.combine_unroll,
            self.combine_count,
            self.combine_weights,
        )


def _fuse_factor(
    want: int | None, wpr: int, vec: int, loads: int, thread_elems: int
) -> int:
    """Adjacent chunks per thread group: as many as the register budget and wpr allow."""
    k = thread_elems // max(1, vec * loads) if want is None else want
    k = max(1, 1 << (max(k, 1).bit_length() - 1))  # floor to a power of two
    return math.gcd(k, max(wpr, 1))


def _loads_per_warp(remaining: int, lanes: int) -> int:
    # Round up so trailing-zero streaming carry spans a power-of-two tree.
    lpw = -(-remaining // lanes)
    return 1 << (lpw - 1).bit_length() if lpw > 1 else lpw


def itree_plan(
    N: int,
    M: int,
    itemsize: int,
    kchunk: int | None = None,
    vmul: int | None = None,
    vec_linear: bool = False,
    stage: bool | None = None,
    device: torch.device | int | str | None = None,
    thread_elems: int | None = None,
    *,
    arch: _ItreeArch | None = None,
) -> _ItreePlan | None:
    """Return the upstream-matching plan, or None to use default order without declining."""
    from .inner_tree_plan import (
        _K_MULTIROW_MAX_LOADS,
        _K_TWO_KERNEL_THRESHOLD,
        _next_power_of_2,
        compute_inner_tree_params,
    )

    if N < 1:
        return None
    kc = kchunk
    # Itemsize-only vec plus identity padding keeps the DAG independent of N divisibility.
    base_vec = tile.TRANSFER_ALIGNMENT // itemsize
    vm = _ITREE_VEC_MUL if vmul is None else vmul
    vec = base_vec * vm
    wle = WARP * vec
    if N <= base_vec * _K_MULTIROW_MAX_LOADS:
        # One fragment, padded to power-of-two loads. Widening base vec would only move bits.
        loads = _next_power_of_2(-(-N // base_vec))
        tm = tile.TileMap(N, itemsize, 1, loads, vec=base_vec)
        return _ItreePlan(
            "multirow",
            vec,
            0,
            _MULTIROW_ROWS_PER_BLOCK,
            _MULTIROW_MAX_DEPTH,
            ((0, N, loads, N),),
            (tm,),
            stage_rows=(
                stage is not False
                and N * itemsize >= _TMA_MIN_STRIDE
                and (arch if arch is not None else _itree_arch(device)).stage_rows
            ),
        )
    prm = compute_inner_tree_params(N, M, vec)
    wpr = prm.num_warps
    if prm.num_batches > _K_TWO_KERNEL_THRESHOLD:
        # One full-batch tile serves both widths; runtime bounds shorten the last batch.
        last = N - (prm.num_batches - 1) * prm.batch_total_elements
        chunk_full = prm.effective_loads * wle
        chunk_last = _loads_per_warp(last, wpr * wle) * wle
        tm = tile.TileMap(
            N,
            itemsize,
            WARP * wpr,
            prm.effective_loads,
            warp_major=True,
            vec=vec,
            exact=False,
        )
        k = _fuse_factor(
            kc,
            wpr,
            vec,
            prm.effective_loads,
            _ITREE_THREAD_ELEMS if thread_elems is None else thread_elems,
        )
        split_stage_rows = (
            arch if arch is not None else _itree_arch(device)
        ).split_stage_rows
        stage_rpb = (
            math.gcd(prm.num_batches, split_stage_rows)
            if (
                stage is not False
                and itemsize <= 4
                and last == prm.batch_total_elements
                and split_stage_rows
            )
            else 0
        )
        stage_rpb = stage_rpb if stage_rpb > 1 else 0
        return _ItreePlan(
            "split",
            vec,
            wpr,
            stage_rpb or 1,
            prm.depth,
            ((0, prm.batch_total_elements, prm.effective_loads, chunk_full),),
            (tm,),
            (
                prm.num_batches,
                prm.batch_total_elements,
                last,
                chunk_full,
                chunk_last,
            ),
            k,
            vec_linear,
            _ITREE_STAGE_E if stage_rpb else 0,
        )
    batches = []
    for b in range(prm.num_batches):
        off = b * prm.batch_total_elements
        remaining = min(prm.batch_total_elements, N - off)
        lpw = _loads_per_warp(remaining, wpr * wle)
        batches.append((off, remaining, lpw, lpw * wle))
    # A batch wholly inside the row is exact; ragged or short final batches retain masks.
    tms = tuple(
        tile.TileMap(
            N,
            itemsize,
            WARP * wpr,
            lpw_b,
            warp_major=True,
            vec=vec,
            exact=off_b + wpr * lpw_b * WARP * vec <= N,
        )
        for (off_b, _rem, lpw_b, _wc) in batches
    )
    k = _fuse_factor(
        kc,
        wpr,
        vec,
        prm.effective_loads,
        _ITREE_THREAD_ELEMS if thread_elems is None else thread_elems,
    )
    rpb = max(1, min(M, _ITREE_BLOCK_THREADS // max(1, WARP * (wpr // k))))
    # Stage one bounded batch only when removing multiple butterflies repays the smem trip.
    span = wpr * prm.effective_loads * WARP * vec
    # Fixed-smem tiles each end in one butterfly over stage_e columns/lane.
    e = min(span // WARP, _ITREE_STAGE_E)
    while e > vec and (span // (e * WARP)) * e * WARP != span:
        e //= 2
    # Require a full run (short: 74.0us versus 51.3us) and multiple butterflies.
    want_stage = (e == _ITREE_STAGE_E and span > WARP * vec) if stage is None else stage
    if (
        want_stage
        and prm.num_batches == 1
        and not vec_linear
        # 128-bit cp.async needs static 16-byte alignment; ragged rows keep register folding.
        and N % vec == 0
        and e % vec == 0
        and e <= tile.MAX_UNROLL
        and (e // vec) & (e // vec - 1) == 0
    ):
        rpb = min(M, prm.rows_per_block) or 1
        k, vec_linear = 1, False
    else:
        e = 0
    return _ItreePlan(
        "looped",
        vec,
        wpr,
        rpb,
        prm.depth,
        tuple(batches),
        tms,
        (),
        k,
        vec_linear,
        e,
    )


def trait_itree_plan(
    trait: Any,
    N: int,
    M: int,
    itemsize: int,
    stage: bool | None = None,
    device: torch.device | int | str | None = None,
    *,
    arch: _ItreeArch | None = None,
) -> _ItreePlan | None:
    """Retune ragged looped plans whose arithmetic or storage favors less fusion."""
    plan = itree_plan(N, M, itemsize, stage=stage, device=device, arch=arch)
    if plan is None or plan.shape != "looped" or all(tm.exact for tm in plan.tms):
        return plan
    from . import traits as T

    if itemsize > 2 or not isinstance(
        trait,
        (T.SumOps, T.NanSumOps, T.ProdOps, T.MeanOps, T.NormOps),
    ):
        return plan
    return itree_plan(
        N,
        M,
        itemsize,
        thread_elems=32,
        stage=stage,
        device=device,
        arch=arch,
    )


def itree_combine_plan(
    itree: _ItreePlan,
    itemsize: int,
    device: torch.device | int | str | None = None,
    nfields: int = 1,
    nrows: int | None = None,
    *,
    arch: _ItreeArch | None = None,
    uniform_count: bool | None = None,
    combine_weights: bool | None = None,
) -> _ItreePlan:
    """Plan the architecture-tuned, bit-neutral fold of each row's split partials."""
    arch = _itree_arch(device) if arch is None else arch
    if uniform_count is None:
        uniform_count = arch.uniform_count
    if combine_weights is None:
        combine_weights = arch.combine_weights
    if arch.combine_unroll < 0:
        raise ValueError("combine_unroll must be nonnegative")
    caps = _hw.caps(device)
    nbatch = itree.split[0]
    rpb = arch.combine_rpb
    if arch.combine_async and nrows is not None:
        # Bucket the staged footprint while retaining a full warp for cp.async.
        live = max(1, min(nrows, rpb))
        rpb = 1 << (live - 1).bit_length()
    grp = arch.combine_group_bytes // itemsize
    g = max(1, math.gcd(nbatch, grp)) if grp else 1
    # A staged tile must DIVIDE the batch count -- a ragged last tile would need predication --
    # so a wide tile does not apply to every nbatch and the walk falls back to narrower ones.
    # Bounded by ELEMENTS, not by copy count: capping on k gave fp64 a 256-element tile where
    # fp32 got 512, so it needed four tiles for the same bytes (9.2us against 7.6).
    tile_n = next(
        (
            k * WARP * g
            for k in (32, 16, 8, 4, 2, 1)
            if k <= arch.combine_async
            and g > 1
            and k * WARP * g <= arch.combine_max
            and nbatch % (k * WARP * g) == 0
            and rpb * (k * WARP * g + g) * itemsize * nfields
            <= caps.smem_per_block_optin
        ),
        0,
    )
    count = (
        itree.split[1]
        if uniform_count
        and tile_n
        and itemsize == 4
        and nfields == 3
        and 0 < nbatch <= 2**24
        and 0 < itree.split[1] <= 2**24
        and itree.split[1] == itree.split[2]
        and not (itree.split[1] & (itree.split[1] - 1))
        else 0
    )
    return _ItreePlan(
        "combine",
        1,
        0,
        rpb,
        itree.depth,
        (),
        (),
        itree.split,
        combine_grp=g,
        combine_tile=tile_n,
        combine_unroll=min(arch.combine_unroll, tile_n),
        combine_count=count,
        combine_weights=bool(count and combine_weights),
    )


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


def _row_args(
    inputs: list[Any],
    outputs: list[Any],
    nchunks: Any,
    nwaves: Any,
    project_n: Any,
) -> tuple[Any, ...]:
    # Row kernels omit the column split and general-axis decode.
    return (
        inputs,
        outputs,
        nchunks,
        nwaves,
        project_n,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        _stream(),
    )


def _launch_itree(
    trait: Any,
    trait_key: str,
    plan: _ItreePlan,
    dt: Any,
    fakes: tuple[Sequence[Any], Sequence[Any]],
    operands: tuple[Sequence[Any], Sequence[Any]],
    N: int,
    tag: str,
    device: torch.device,
    nouts: int = 1,
    dsts: Sequence[torch.dtype] = (),
    align: int = 0,
    compact: bool = True,
) -> None:
    """Launch one stage, keying the input layout and baked alignment."""
    op = tile.TileReduce(
        trait,
        dt,
        "row",
        N,
        nouts=nouts,
        final=plan.shape != "split",
        order="inner_tree",
        itree=plan,
    )

    # Destination types are baked, so include them to prevent a wrong cached plan.
    key = (
        tag,
        trait_key,
        dt,
        tuple(dsts),
        align,
        compact,
        str(device),
    ) + op.cache_sig
    launch = (Int32(N // plan.vec), None, Int32(N))
    args = lambda pair: _row_args(  # noqa: E731
        pair[0], pair[1], launch[0], launch[1], launch[2]
    )
    build = lambda: _compile(op, *args(fakes))  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(*args(operands))


def _run_itree(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    itree: _ItreePlan,
    nouts: int = 1,
    out: Sequence[torch.Tensor] | None = None,
    *,
    arch: _ItreeArch | None = None,
) -> tuple[torch.Tensor, ...]:
    """Launch each stage; split buffers are per field and supplied outputs are 1-D unit-stride."""
    M, N = x.shape

    def results():
        if out is not None:
            return list(out)
        return [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes[:nouts]]

    dt = torch2cute[x.dtype]
    # Storage offsets may underalign; declare and key the supported width.
    itemsize = x.element_size()
    natural = tile.align_bytes(N, itemsize)
    align = _L.supported_alignment(x, natural)
    compact = x.stride(0) == N
    if not compact:
        align = min(align, math.gcd(x.stride(0) * itemsize, natural))
    if align < natural and itree.stage_e:
        # cp.async's 128-bit atom needs a statically `natural`-aligned source, so a misaligned base
        # cannot use the staged form at all -- it fails IR verification rather than the load. Serve
        # this call with the UNSTAGED form of the same plan: staging is bit-neutral, so the DAG and
        # therefore the result do not move (verified bitwise against the reference at align 8 and 4
        # for every staged shape).
        itree = cast(
            _ItreePlan,
            itree_plan(
                N,
                M,
                x.element_size(),
                kchunk=itree.kchunk,
                stage=False,
                device=x.device,
                arch=arch,
            ),
        )
    # N is baked into the DAG, so the row extent is static and only M rides in dynamically.
    if compact:
        fake_in = _L.fake_compact(dt, (_L.sym(), N), stride_order=(1, 0), align=align)
    else:
        fake_in = cute.runtime.make_fake_tensor(
            dt,
            (_L.sym(), N),
            (cute.sym_int64(divisibility=align // itemsize), 1),
            assumed_align=align,
        )
    fake_1d = lambda t, a=None: _L.fake_compact(  # noqa: E731
        torch2cute[t.dtype], (_L.sym(),), align=a
    )
    if itree.shape != "split":
        outs = results()
        _launch_itree(
            trait,
            trait_key,
            itree,
            dt,
            ([fake_in], [fake_1d(o) for o in outs]),
            ([_L.read_only(x)], list(outs)),
            N,
            "rowitree",
            x.device,
            nouts,
            tuple(o.dtype for o in outs),
            align,
            compact,
        )
        return tuple(outs)
    # Split writes one field-typed partial per (row, batch), then folds them linearly.
    nbatch = itree.split[0]
    parts = [
        torch.empty(M * nbatch, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    _launch_itree(
        trait,
        trait_key,
        itree,
        dt,
        ([fake_in], [fake_1d(p) for p in parts]),
        ([_L.read_only(x)], list(parts)),
        N,
        "rowitree1",
        x.device,
        nouts,
        tuple(p.dtype for p in parts),
        align,
        compact,
    )
    outs = results()
    # DECLARE the partials' alignment or stage 2 reads them one element at a time: autovec_copy
    # cannot widen past what the descriptor claims. Keyed on the BATCH COUNT, a row's stride.
    palign = tile.align_bytes(nbatch, parts[0].element_size())
    _launch_itree(
        trait,
        trait_key,
        itree_combine_plan(
            itree,
            parts[0].element_size(),
            x.device,
            nfields=trait.nfields,
            nrows=M,
            arch=arch,
        ),
        dt,
        ([fake_1d(p, palign) for p in parts], [fake_1d(o) for o in outs]),
        ([_L.read_only(p) for p in parts], list(outs)),
        N,
        "rowitree2",
        x.device,
        nouts,
        tuple(o.dtype for o in outs),
        palign,
    )
    return tuple(outs)


def reduce_row_itree(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Write 2-D `x` rows to 1-D `out`; return False only when no inner-tree plan exists."""
    M, N = x.shape
    arch = _full_itree_arch(trait, trait_key, x, [out.dtype])
    itree = trait_itree_plan(trait, N, M, x.element_size(), device=x.device, arch=arch)
    if itree is None:
        return False
    _run_itree(trait, trait_key, x, [out.dtype], itree, out=[out], arch=arch)
    return True


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
    order: Literal["linear", "inner_tree"] | None = None,
    ragged_vector: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Reduce 2-D `x` rows, returning outputs or raw field partials when `final=False`."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    if M >= _INT32_LIMIT or N >= _INT32_LIMIT:
        raise AssertionError(f"row reduction needs M and N < 2^31, got M={M}, N={N}")
    # leaf/combine serves every trait and N. The opt-in gate leaves partial stages and
    # explicit launch shapes on the default order; explicit requests raise below.
    if order not in (None, "linear", "inner_tree"):
        raise ValueError(f"order must be None, 'linear' or 'inner_tree', got {order!r}")
    if ragged_vector and order != "linear":
        raise ValueError("ragged vectors require explicit order='linear'")
    itree, arch = None, None
    if (order == "inner_tree" or (order is None and inner_tree_order_enabled())) and (
        final and threads_per_row is None
    ):
        arch = _full_itree_arch(trait, trait_key, x, out_dtypes[:nouts])
        itree = trait_itree_plan(
            trait, N, M, x.element_size(), device=x.device, arch=arch
        )
    if order == "inner_tree" and itree is None:
        # Explicit inner_tree cannot silently use another DAG; only the env gate may fall back.
        raise ValueError(
            f"order='inner_tree' cannot be honoured here: {final=} {threads_per_row=} "
            f"plan={trait_itree_plan(trait, N, M, x.element_size(), device=x.device) is not None}"
        )
    if itree is not None:
        return _run_itree(trait, trait_key, x, out_dtypes, itree, nouts, arch=arch)
    cfg = row_config(N, x.element_size() * 8)
    # Scalar rows use 16 to hide narrow-load latency; vectorized rows use 4, at or near
    # the measured optimum across 4/8/16/32.
    if unroll is None:
        unroll = (
            4 if ragged_vector else 16 if tile.vec_size(N, x.element_size()) == 1 else 4
        )
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
        vec=128 // (isz * 8) if ragged_vector else None,
        ragged_vector=ragged_vector,
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
        else _L.supported_alignment(x, tile.TRANSFER_ALIGNMENT)
        if ragged_vector
        else _L.supported_alignment(x, tile.align_bytes(N, isz))
    )

    def _fake():
        # TMA bakes N; runtime folds share a vector class.
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
        elif ragged_vector:
            fake_in = _L.fake_compact(
                dt,
                (_L.sym(), N),
                stride_order=(1, 0),
                align=align,
            )
        else:
            fake_in = _L.fake_compact(
                dt,
                (_L.sym(), _L.sym(op.vec)),
                stride_order=(1, 0),
                align=align,
            )
        return _row_args(
            [fake_in],
            [_L.fake_compact(torch2cute[o.dtype], (_L.sym(),)) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
        )

    # Pointer-dependent alignment is compiled, so include it in the key.
    dts = tuple(out_dtypes[:ndst])
    key = (
        "rowtile",
        trait_key,
        x.dtype,
        dts,
        align,
        str(x.device),
    ) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    fn(*_row_args([_L.read_only(x)], list(outs), nchunks, nwaves, Int32(N)))
    return tuple(outs)
