# Reduction dispatcher and general-axis plan for the shared TileReduce kernel.
# TensorIterator geometry handles every layout. Runtime values keep kernel count
# O(op * dtype * pair-count); cache_sig compiles decode depths, outputs, index mode,
# projection/partial mode, and fold clamps.

import math
from collections.abc import Sequence
from typing import Any, cast, Literal, TYPE_CHECKING

from cutlass import Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import _storage, tile
from .traits import WARP


if TYPE_CHECKING:
    from .kernel_rowtile import _ItreePlan


# (extent, element-stride) pairs from TensorIterator, fastest dim first.
Pairs = Sequence[tuple[int, int]]
_INT32_LIMIT = 1 << 31
_GRID_X_MAX = _INT32_LIMIT - 1


class ReduceBlock:
    def __init__(
        self,
        trait: Any,
        *,
        count: int,
        num_o: int,
        red_pairs: Sequence[tuple[int, int]],
        kept_pairs: Sequence[tuple[int, int]],
        in_base: int = 0,
        limit: int | None = None,
        project_n: int | None = None,
        nouts: int = 1,
        final: bool = True,
        gidx_from: Literal["r", "flat", "chunk"] = "r",
        flat_tail: bool = False,
        ragged_chunk: bool = False,
        from_partials: bool = False,
        block: int = 128,
        order: Literal["linear", "inner_tree"] = "linear",
        itree: Any = None,
    ) -> None:
        self.trait = trait
        self.count = count  # elements reduced per output (= prod red exts)
        self.num_o = num_o  # number of outputs / blocks (= prod kept exts)
        # An empty reduced list is LEGAL: an extent-1 reduced axis coalesces away in TI, leaving a
        # fold of one element per output. An empty KEPT list is a full reduction.
        self.red_pairs = tuple(red_pairs)
        self.kept_pairs = tuple(kept_pairs)
        self.npairs_red = len(self.red_pairs)
        self.npairs_kept = len(self.kept_pairs)
        self.wide_count = count >= _INT32_LIMIT
        self.wide_output = num_o >= _INT32_LIMIT
        self.wide_gidx = getattr(trait, "has_index", False) and trait.idx is Int64
        self.wide_red = any(ext >= _INT32_LIMIT for ext, _ in self.red_pairs)
        self.wide_kept = any(ext >= _INT32_LIMIT for ext, _ in self.kept_pairs)
        self.in_base = in_base  # flat input offset of output coordinate 0
        self.limit = limit if limit is not None else count  # ragged tail bound
        # Projection divisor: count normally, or original L when stage 2 folds G partials.
        self.project_n = project_n if project_n is not None else count
        self.nouts = nouts
        self.final = final
        self.gidx_from = gidx_from  # "r", reduce-all "flat", or row-split "chunk"
        self.flat_tail = flat_tail  # clamp reduce-all stage 1 to limit
        self.ragged_chunk = ragged_chunk  # clamp each output's reduced run
        self.from_partials = from_partials
        self.block = block
        # Plan one general-axis block per output; the shared body performs mixed-radix folding.
        self.tile = tile.TileReduce(
            trait,
            None,
            "general",
            0,
            threads_per_block=block,
            nouts=nouts,
            final=final,
            combine=from_partials,
            npairs_red=self.npairs_red,
            npairs_kept=self.npairs_kept,
            wide_count=self.wide_count,
            wide_output=self.wide_output,
            wide_gidx=self.wide_gidx,
            wide_red=self.wide_red,
            wide_kept=self.wide_kept,
            gidx_from=gidx_from,
            flat_tail=flat_tail,
            ragged_chunk=ragged_chunk,
            order=order,
            itree=itree,
        )

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        # Derive the key from every const_expr baked by the shared body.
        return self.tile.cache_sig

    @property
    def geom_sig(self) -> tuple[Any, ...]:
        # Cache boxed runtime geometry (~6us) separately while sharing the structural kernel.
        return (
            self.count,
            self.num_o if self.wide_output else None,
            self.red_pairs,
            self.kept_pairs,
            self.in_base,
            self.limit,
            self.project_n,
        )


# Host-side geometry and plans; no kernels below.
_stream = _L.stream
# _L.compile_kernel uses fake operands and tvm-ffi, avoiding per-call tensor wrapping.
_compile = _L.compile_kernel
_COMPILE_CACHE = {}  # structural key -> compiled kernel (one per cache_sig)
_PLAN = {}  # (structural key, geom_sig) -> (compiled fn, pre-boxed geometry args)


def _fakes(ts: Sequence[torch.Tensor], *, int64_extent: bool = False) -> list[Any]:
    # Dynamic flat descriptors let one structural kernel serve every length.
    return [
        _L.fake_compact(
            torch2cute[t.dtype],
            (_L.sym_int64() if int64_extent else _L.sym(),),
        )
        for t in ts
    ]


def _operands(ts: Sequence[torch.Tensor], read_only: bool = False) -> list[Any]:
    # Pass real tensors; read_only() prevents COW input materialization during export.
    return [_L.read_only(t) for t in ts] if read_only else list(ts)


def _exts(pairs: Pairs, wide: bool = False) -> list[Any] | None:
    # FastDivmod encodes Int32 divisors; wider dimensions use native Int64 division.
    dtype = Int64 if wide else Int32
    return [dtype(ext) for ext, _ in pairs] or None


def _strides(pairs: Pairs) -> list[Any] | None:
    return [Int64(strd) for _, strd in pairs] or None


def _geom_args(op: ReduceBlock) -> tuple[Any, ...]:
    # Runtime geometry: both decodes' extents/strides and scalar bounds.
    grid_x = min(op.num_o, _GRID_X_MAX) if op.wide_output else None
    return (
        (Int64 if op.wide_count else Int32)(op.count),
        None,
        Int64(op.project_n),
        None if grid_x is None else Int32(grid_x),
        None if grid_x is None else Int32(-(-op.num_o // grid_x)),
        _exts(op.red_pairs, op.wide_red),
        _strides(op.red_pairs),
        _exts(op.kept_pairs, op.wide_kept),
        _strides(op.kept_pairs),
        Int64(op.in_base),
        Int64(op.limit),
    )


def _launch(
    op: ReduceBlock,
    key: tuple[Any, ...],
    ins: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
) -> None:
    # _PLAN caches boxed geometry (~6us); _COMPILE_CACHE deduplicates structural kernels.
    op_name = key[1]
    key = (str(ins[0].device),) + key
    plan = _PLAN.get((key, op.geom_sig))
    if plan is None:
        fn = cached_plan(
            _COMPILE_CACHE,
            key,
            lambda: _compile(
                op.tile,
                # A flat input storage span can exceed 2**31 even though each
                # decoded kept and reduced extent fits Int32.
                _fakes(ins, int64_extent=True),
                _fakes(
                    outs,
                    int64_extent=op.wide_output
                    or (
                        op.final and any(width == 2 for width in op.tile.output_widths)
                    ),
                ),
                *_geom_args(op),
                _stream(),
            ),
            op=f"aten::{op_name}",
        )
        plan = (fn, _geom_args(op))
        _PLAN[(key, op.geom_sig)] = plan
    fn, geom = plan
    fn(_operands(ins, read_only=True), _operands(outs), *geom, _stream())


def _ti_pairs(x: torch.Tensor, out: torch.Tensor) -> tuple[Pairs, Pairs]:
    """Return reduced and kept (extent, input-stride) pairs from TensorIterator.
    Zero output stride marks reductions; kept dimensions sort by output stride for
    fastest-first block decoding. Reduced pairs retain TensorIterator order.
    """
    it = reduce_op(out, x)
    in_str = it.element_strides(it.noutputs)  # input operand follows the outputs
    out_str = it.element_strides(0)
    red = [(it.shape[i], in_str[i]) for i in range(it.ndim) if out_str[i] == 0]
    kept = [
        (it.shape[i], in_str[i], out_str[i]) for i in range(it.ndim) if out_str[i] != 0
    ]
    kept.sort(key=lambda p: p[2])  # fastest output dim first
    return red, [(e, s) for e, s, _ in kept]


def _probe(x: torch.Tensor, red_axes: set[int]) -> torch.Tensor:
    # Give reduce_op an independent output with reduced dimensions set to one. Sharing this
    # decode keeps classification and fallback consistent; a view of x would materialize COW.
    return torch.empty(
        [1 if i in red_axes else s for i, s in enumerate(x.shape)],
        device=x.device,
        dtype=x.dtype,
    )


def _flat(x: torch.Tensor) -> torch.Tensor:
    # View all storage at stride one because TI strides are storage-relative; reshape could copy.
    storage_view = _storage.real_view(x)
    n = max(
        storage_view.untyped_storage().nbytes() // storage_view.element_size(),
        1,
    )
    return torch.as_strided(storage_view, (n,), (1,), storage_offset=0)


def _kernel_out(out: torch.Tensor) -> torch.Tensor:
    return _storage.flat_view(out)


def _kernel_outs(outs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [_kernel_out(out) for out in outs]


# Fast-path classification shared by routing and condition gates, after TI coalescing.


def fast_kind(red_pairs: Pairs, kept_pairs: Pairs, nouts: int) -> str | None:
    """Select row or single-output column reduction by a dense 2D TI view's stride-one pair."""
    if len(kept_pairs) == 0:
        return "all"
    if len(red_pairs) != 1 or len(kept_pairs) != 1:
        return None
    if red_pairs[0][1] == 1:  # reduced run is innermost/contiguous -> row
        return "row"
    if kept_pairs[0][1] == 1 and nouts == 1:  # kept innermost -> col
        return "col"
    return None


# Largest one-block register-loaded row; only merging uses smem. Larger uses multi-CTA.
_MAX_ROW_BYTES = 192 * 1024
# Bound loads when odd or prime N collapses vector width. 64 separates measured wins from
# losses; routing beyond it improved 0.08-0.17x to 1.93-2.41x of ATen.
_ONESHOT_MAX_LOADS = 64
# Cap ragged chunks per row and therefore the stage-2 fold.
_C_MAX_ROW = 64

# General-axis occupancy baselines, not a tuned performance surface.
_K0_BLOCK = 128
_K0_ALL_BLOCK = 256
_K0_ALL_GRID_MULT = 4


def _oneshot_ok(x: torch.Tensor) -> bool:
    # Require both the row-size and per-thread-load bounds.
    N = x.shape[-1]
    if N * x.element_size() > _MAX_ROW_BYTES:
        return False
    from . import kernel_rowtile as rt

    width = x.element_size() * 8
    vec = math.gcd(N, 128 // width)
    threads_per_row = max(WARP, rt.row_config(N, width).threads_per_row)
    return -(-N // (threads_per_row * vec)) <= _ONESHOT_MAX_LOADS


def _try_fast_row(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
) -> tuple[torch.Tensor, ...] | None:
    # Fast contiguous 2D last-dimension path. It needs no index remap, so index traits work.
    if x.dim() != 2 or x.stride(-1) != 1:
        return None
    N = x.shape[-1]
    if N < 1:
        return None
    if nouts not in (1, 2):
        return None
    from . import kernel_rowtile as rt

    # Packed rows floor at one warp (25% utilized at N=32), but inner-tree order owns
    # its thread map. Forcing threads_per_row=1 changed bits at (524288, 16) and (524288, 128).
    if (
        rt.narrow_row(N, x.element_size(), x.shape[0])
        and not rt.inner_tree_order_enabled()
    ):
        return rt.reduce_row_tile(
            trait, trait_key, x, out_dtypes, nouts=nouts, threads_per_row=1
        )
    if _oneshot_ok(x):
        return rt.reduce_row_tile(trait, trait_key, x, out_dtypes, nouts=nouts)
    # Bypass default-order xcta whenever the fold gate has a plan. Otherwise bits differed at
    # (64, 100000), (8, 200000), and (8, 1000000).
    if rt.inner_tree_order_enabled() and (
        rt.itree_plan(N, x.shape[0], x.element_size()) is not None
    ):
        return rt.reduce_row_tile(trait, trait_key, x, out_dtypes, nouts=nouts)
    from . import kernel_xcta as xc

    if nouts == 2:
        # Project both fields from one fused split; one-block-per-row measured
        # 0.63x of ATen at N=65536 and 0.20x at 131072.
        res = xc.reduce_row_xcta_2out(trait, trait_key, x, out_dtypes)
        if res is not None:
            return res
        # Fall back when N has no exact divisor split.
        return _two_stage_row(trait, trait_key, x, out_dtypes, nouts)
    res = xc.reduce_row_xcta(trait, trait_key, x, out_dtypes[0])
    if res is not None:
        return (res,)
    # Ragged splitting handles prime N and absolute indices without exact divisibility.
    return _two_stage_row(trait, trait_key, x, out_dtypes, nouts)


def _as_shape(out: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
    # Resize rather than view: ATen reductions allocate non-aliasing outputs.
    if tuple(out.shape) == tuple(out_shape):
        return out
    reshaped = out.reshape(out_shape)
    if out._base is not None:
        return reshaped.clone()
    if reshaped._base is None:
        return reshaped
    out.resize_(out_shape)
    return out


def _two_stage_row(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    block: int = _K0_ALL_BLOCK,
) -> tuple[torch.Tensor, ...] | None:
    # Ragged chunks handle N without an in-window divisor; stage 1 clamps row tails and
    # carries global indices, preserving first-wins ties. Prime (8, 131071) measured
    # 0.28x of ATen without this split. Decline C == 1.
    M, N = x.shape
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    # Fill the device, then align chunk bases to 16 bytes; C follows from chunk size.
    C = max(1, min(_C_MAX_ROW, -(-(sm * _K0_ALL_GRID_MULT) // max(1, M))))
    if C == 1:
        return None
    vec = max(1, 16 // x.element_size())
    s_chunk = max(vec, -(-N // C) // vec * vec)
    C = -(-N // s_chunk)
    if C == 1:
        return None

    parts = [
        torch.empty(M * C, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1 emits each (row, chunk); the leading kept pair identifies the chunk.
    s1 = ReduceBlock(
        trait,
        count=s_chunk,
        num_o=M * C,
        red_pairs=[(s_chunk, 1)],
        kept_pairs=[(C, s_chunk), (M, N)],
        limit=N,
        ragged_chunk=True,
        gidx_from="chunk" if getattr(trait, "has_index", False) else "r",
        nouts=trait.nfields,
        final=False,
        block=block,
    )
    _launch(s1, ("rowrag1", trait_key, x.dtype) + s1.cache_sig, [_flat(x)], parts)

    # Stage 2 folds C partials and projects once with the true row length.
    s2 = ReduceBlock(
        trait,
        count=C,
        num_o=M,
        red_pairs=[(C, 1)],
        kept_pairs=[(M, C)],
        from_partials=True,
        project_n=N,
        nouts=nouts,
        final=True,
        block=block,
    )
    _launch(s2, ("rowrag2", trait_key, tuple(out_dtypes)) + s2.cache_sig, parts, outs)
    return tuple(outs)


def _indexed_itree_plan(
    count: int,
    num_o: int,
    itemsize: int,
    device: torch.device,
) -> "_ItreePlan | None":
    from . import kernel_rowtile as rt

    plan = rt.itree_plan(count, num_o, itemsize, stage=False)
    if plan is None:
        return None
    return plan


def _try_indexed_itree(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    red_pairs: Pairs,
    kept_pairs: Pairs,
    num_o: int,
    count: int,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
) -> tuple[torch.Tensor, ...] | None:
    """Apply the fixed logical-row DAG through arbitrary storage strides."""
    from . import kernel_rowtile as rt

    plan = _indexed_itree_plan(count, num_o, x.element_size(), x.device)
    if plan is None:
        return None
    nbatch = plan.split[0] if plan.shape == "split" else 1
    if num_o * nbatch >= 2**31:
        return None

    if plan.shape != "split":
        outs = [torch.empty(num_o, device=x.device, dtype=d) for d in out_dtypes]
        op = ReduceBlock(
            trait,
            count=count,
            num_o=num_o,
            red_pairs=red_pairs,
            kept_pairs=kept_pairs,
            in_base=int(x.storage_offset()),
            nouts=nouts,
            order="inner_tree",
            itree=plan,
        )
        key = ("genitree", trait_key, x.dtype, tuple(out_dtypes)) + op.cache_sig
        _launch(op, key, [_flat(x)], _kernel_outs(outs))
        return tuple(outs)

    parts = [
        torch.empty(
            num_o * nbatch,
            device=x.device,
            dtype=cute2torch[trait.fdtypes[f]],
        )
        for f in range(trait.nfields)
    ]
    s1 = ReduceBlock(
        trait,
        count=count,
        num_o=num_o,
        red_pairs=red_pairs,
        kept_pairs=kept_pairs,
        in_base=int(x.storage_offset()),
        nouts=trait.nfields,
        final=False,
        order="inner_tree",
        itree=plan,
    )
    key1 = ("genitree1", trait_key, x.dtype) + s1.cache_sig
    _launch(s1, key1, [_flat(x)], parts)

    outs = [torch.empty(num_o, device=x.device, dtype=d) for d in out_dtypes]
    combine = rt.itree_combine_plan(plan)
    s2 = ReduceBlock(
        trait,
        count=nbatch,
        num_o=num_o,
        red_pairs=[],
        kept_pairs=[],
        project_n=count,
        nouts=nouts,
        order="inner_tree",
        itree=combine,
    )
    part_dtypes = tuple(p.dtype for p in parts)
    key2 = (
        "genitree2",
        trait_key,
        tuple(out_dtypes),
        part_dtypes,
    ) + s2.cache_sig
    _launch(s2, key2, parts, _kernel_outs(outs))
    return tuple(outs)


def _reduce(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    dims: int | Sequence[int] | None,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    block: int = _K0_BLOCK,
) -> tuple[torch.Tensor, ...]:
    # TI drives all dimensions and layouts through this path; return nouts tensors.
    if not x.is_cuda:
        raise AssertionError(f"need a CUDA input, got {x.device}")
    red_axes = (
        range(x.dim()) if dims is None else ([dims] if isinstance(dims, int) else dims)
    )
    red_axes = {d % x.dim() for d in red_axes}
    out_shape = [s for i, s in enumerate(x.shape) if i not in red_axes]
    num_o = max(1, math.prod(out_shape))
    count = x.numel() // num_o
    red_pairs, kept_pairs = _ti_pairs(x, _probe(x, red_axes))

    from . import kernel_rowtile as rt

    if (
        not getattr(trait, "complex_input", False)
        and rt.inner_tree_order_enabled()
        and count < _INT32_LIMIT
        and num_o < _INT32_LIMIT
    ):
        kind = fast_kind(red_pairs, kept_pairs, nouts)
        if x.is_contiguous() and kind in ("row", "all"):
            ordered = rt.reduce_row_tile(
                trait,
                trait_key,
                x.reshape(num_o, count),
                out_dtypes,
                nouts=nouts,
                order="inner_tree",
            )
        else:
            ordered = _try_indexed_itree(
                trait,
                trait_key,
                x,
                red_pairs,
                kept_pairs,
                num_o,
                count,
                out_dtypes,
                nouts,
            )
        if ordered is not None:
            return tuple(_as_shape(o, out_shape) for o in ordered)

    # One-output reductions use reduce_all's split instead of one block for the entire row.
    # This includes full dims and M=1 rows whose size-one kept axes TI removes.
    if math.prod(out_shape) == 1 and nouts == 1 and x.is_contiguous():
        out = reduce_all(trait, trait_key, x, out_dtypes[0], block=block)
        return (_as_shape(out, out_shape),)

    # Reshape post-TI contiguous innermost reductions onto a fast kernel; general remains
    # the fallback for direct calls and declines.
    if len(out_shape) > 0 and x.is_contiguous():
        kind = fast_kind(red_pairs, kept_pairs, nouts)
        red_n = x.numel() // max(1, math.prod(out_shape))
        if kind == "row":
            x2 = x.reshape(math.prod(out_shape), red_n)
            fast = _try_fast_row(trait, trait_key, x2, out_dtypes, nouts)
            if fast is not None:
                return tuple(o.reshape(out_shape) for o in fast)
        elif kind == "col":
            # Splitting the reduced axis supplies parallelism for tall-narrow inputs:
            # 7.24x, 2.53x, and 1.49x of ATen at (65536, 256), (16384, 1024), and (4096, 4096).
            from . import kernel_coltile as ct

            x2 = x.reshape(red_n, math.prod(out_shape))
            out = ct.reduce_col_tile(trait, trait_key, x2, out_dtypes[0])
            return (_as_shape(out, out_shape),)

    outs = [torch.empty(out_shape, device=x.device, dtype=d) for d in out_dtypes]
    op = ReduceBlock(
        trait,
        count=count,
        num_o=num_o,
        red_pairs=red_pairs,
        kept_pairs=kept_pairs,
        in_base=cast(int, x.storage_offset()),
        nouts=nouts,
        block=block,
    )
    key = ("reduce", trait_key, x.dtype, tuple(out_dtypes)) + op.cache_sig
    _launch(op, key, [_flat(x)], [o.reshape(-1) for o in outs])
    return tuple(outs)


def reduce_dim(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    dims: int | Sequence[int] | None,
    out_dtype: torch.dtype,
    block: int = _K0_BLOCK,
) -> torch.Tensor:
    return _reduce(trait, trait_key, x, dims, [out_dtype], 1, block=block)[0]


def reduce_dim2(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    dims: int | Sequence[int] | None,
    out_dtypes: Sequence[torch.dtype],
    block: int = _K0_BLOCK,
) -> tuple[torch.Tensor, ...]:
    return _reduce(trait, trait_key, x, dims, list(out_dtypes), 2, block=block)


def _grid_size(L: int, block: int, sm_count: int, grid_mult: int = 4) -> int:
    # Choose G chunks to fill grid_mult waves, trading stage-1 parallelism for stage-2 work.
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtype: torch.dtype,
    block: int = _K0_ALL_BLOCK,
    grid_mult: int = _K0_ALL_GRID_MULT,
) -> torch.Tensor:
    return _reduce_all(trait, trait_key, x, [out_dtype], 1, block, grid_mult)[0]


def _reduce_all(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    block: int,
    grid_mult: int,
) -> tuple[torch.Tensor, ...]:
    # Try the one-shot row kernel, fused cross-CTA split, then grid-striding fallback.
    # All preserve flat indices because reduce-all is a single row.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"reduce-all needs a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    L = x.numel()
    xf = x.reshape(-1)
    # Keep one-shot inputs out of xcta, which adds a ~1.9us launch or declines them.
    # Direct routing measured 1.2-2.1x over ATen.
    x2 = xf.view(1, -1)
    if _oneshot_ok(x2):
        from . import kernel_rowtile as rt

        # Avoid row-packing threads for a single-row launch; None keeps the existing config.
        cfg = rt.single_row_config(L, x.element_size() * 8)
        outs = rt.reduce_row_tile(
            trait,
            trait_key,
            x2,
            out_dtypes,
            nouts=nouts,
            threads_per_row=None if cfg is None else cfg.threads_per_row,
            threads_per_block=None if cfg is None else cfg.threads_per_block,
        )
        return tuple(_as_shape(o, ()) for o in outs)
    from . import kernel_xcta as xc

    if nouts == 1:
        res = xc.reduce_row_xcta(trait, trait_key, xf, out_dtypes[0], flatten=True)
        res = None if res is None else (res,)
    else:
        res = xc.reduce_row_xcta_2out(trait, trait_key, xf, out_dtypes, flatten=True)
    if res is not None:
        return res
    # If xcta declines, grid-stride any L without reshaping or extent-dependent compile cost.
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(1, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1 models G contiguous chunks as kept (G, chunk), reduced (chunk, 1);
    # flat_tail guards the last and gidx_from="flat" preserves global indices.
    s1 = ReduceBlock(
        trait,
        count=chunk,
        num_o=G,
        red_pairs=[(chunk, 1)],
        kept_pairs=[(G, chunk)],
        limit=L,
        flat_tail=True,
        gidx_from="flat",
        nouts=trait.nfields,
        final=False,
        block=block,
    )
    _launch(s1, ("all1", trait_key, x.dtype) + s1.cache_sig, [xf], parts)

    # Stage 2 folds G partials and projects with the true element count, keyed in cache_sig.
    s2 = ReduceBlock(
        trait,
        count=G,
        num_o=1,
        red_pairs=[(G, 1)],
        kept_pairs=[],
        from_partials=True,
        project_n=L,
        nouts=nouts,
        final=True,
        block=block,
    )
    _launch(s2, ("all2", trait_key, tuple(out_dtypes)) + s2.cache_sig, parts, outs)
    return tuple(_as_shape(o, ()) for o in outs)
