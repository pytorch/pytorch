# Reduction dispatcher and general-axis plan for the shared TileReduce kernel.
# TensorIterator geometry handles every layout. Runtime values keep kernel count
# O(op * dtype * pair-count); cache_sig compiles decode depths, outputs, index mode,
# projection/partial mode, and fold clamps.

import math
from collections.abc import Sequence
from typing import Any, cast, Literal, NamedTuple, TYPE_CHECKING

from cutlass import Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import _storage, tile
from .traits import WARP, welford_nouts


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
        tree_count: int = 0,
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
            general_tree_count=tree_count,
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


def _fakes(
    ts: Sequence[torch.Tensor],
    *,
    int64_extent: bool = False,
    alignments: Sequence[int] | None = None,
) -> list[Any]:
    # Dynamic flat descriptors let one structural kernel serve every length.
    return [
        _L.fake_compact(
            torch2cute[t.dtype],
            (_L.sym_int64() if int64_extent else _L.sym(),),
            align=None if alignments is None else alignments[i],
        )
        for i, t in enumerate(ts)
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
    alignments = None
    itree = op.tile.itree
    if (
        op.tile.order == "inner_tree"
        and itree.shape == "combine"
        and itree.combine_tile
    ):
        # Async atoms require the partial buffers' proven alignment in the fake descriptors.
        alignments = tuple(itree.combine_grp * t.element_size() for t in ins)
        if any(
            _L.supported_alignment(t, align) < align
            for t, align in zip(ins, alignments)
        ):
            raise ValueError(
                "inner-tree async combine requires aligned partial buffers"
            )
        key += (("input_alignments", alignments),)
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
                _fakes(ins, int64_extent=True, alignments=alignments),
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
    """Select row or column reduction by a dense 2D TI view's stride-one pair."""
    if len(kept_pairs) == 0:
        return "all"
    if len(red_pairs) != 1 or len(kept_pairs) != 1:
        return None
    if red_pairs[0][1] == 1:  # reduced run is innermost/contiguous -> row
        return "row"
    if kept_pairs[0][1] == 1 and nouts in (1, 2):  # kept innermost -> col
        return "col"
    return None


def _physical_col_view(
    x: torch.Tensor,
    red_axes: set[int],
    red_pairs: Pairs,
    kept_pairs: Pairs,
    count: int,
    num_o: int,
) -> torch.Tensor | None:
    if red_pairs == [(count, num_o)] and kept_pairs == [(num_o, 1)]:
        return torch.as_strided(
            x,
            (count, num_o),
            (num_o, 1),
            storage_offset=x.storage_offset(),
        )
    dims = sorted(red_axes)
    if (
        not x.is_contiguous()
        or not dims
        or dims != list(range(dims[0], dims[-1] + 1))
        or dims[0] == 0
        or dims[-1] + 1 == x.dim()
    ):
        return None
    B = math.prod(x.shape[: dims[0]])
    R = math.prod(x.shape[dims[0] : dims[-1] + 1])
    C = math.prod(x.shape[dims[-1] + 1 :])
    return x.reshape(B, R, C)


# Largest one-block register-loaded row; only merging uses smem. Larger uses multi-CTA.
_MAX_ROW_BYTES = 192 * 1024
# Bound loads when odd or prime N collapses vector width. Width 4097 needs 65 scalar
# loads for 16-bit inputs; keeping it on rowtile measured 41us instead of 102us.
_ONESHOT_MAX_LOADS = 65
# Minimum reduced elements per output before a cross-CTA split is worth its second launch.
_SPLIT_MIN_COUNT = 256
_WIDE_SPLIT_MIN_BYTES = 32 * 1024 * 1024

# General-axis occupancy baselines, not a tuned performance surface.
_K0_BLOCK = 128
_K0_ALL_BLOCK = 256
_K0_ALL_GRID_MULT = 4


class _AllConfig(NamedTuple):
    block: int = _K0_ALL_BLOCK
    grid_mult: int = _K0_ALL_GRID_MULT


def select_all_config(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    count: int,
    *,
    nfields: int,
    nouts: int,
) -> _AllConfig:
    size = count * dtype.itemsize
    if (
        cc == (10, 0)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and size in (256 << 20, 2 << 30)
        and nfields == 2
        and nouts in (1, 2)
        and trait_key.startswith(("argmax", "argmin", "max.dim", "min.dim"))
    ):
        return _AllConfig(block=512, grid_mult=8)
    return _AllConfig()


class _GeneralConfig(NamedTuple):
    block: int = _K0_BLOCK
    kernel_order: Literal["linear", "inner_tree"] = "linear"
    rule: str = "general_default"
    uniform_tree: bool = False


def select_general_config(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    count: int,
    num_o: int,
    red_pairs: Pairs,
    kept_pairs: Pairs,
    *,
    order: Literal["unordered", "inner_tree"],
    nfields: int,
    nouts: int,
    acc_bits: int = 32,
    alignment: int = 16,
) -> _GeneralConfig:
    if order == "inner_tree":
        return _GeneralConfig(kernel_order="inner_tree", rule="required_inner_tree")
    if order != "unordered":
        raise ValueError(f"unknown reduction order: {order!r}")
    strided_row = (
        cc == (10, 0)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and tuple(red_pairs) == ((count, 2),)
        and tuple(kept_pairs) == ((num_o, 2 * count),)
        and acc_bits == 32
        and alignment >= 16
    )
    if (
        strided_row
        and count in (256, 257)
        and (
            nfields == 1
            and nouts == 1
            and trait_key in ("sum", "mean", "amax", "vnorm2")
            or nfields == 2
            and nouts == 1
            and trait_key.startswith(("argmax", "argmin"))
            or nfields == 3
            and welford_nouts(trait_key) == nouts
        )
    ):
        return _GeneralConfig(kernel_order="inner_tree", rule="b200_strided_c256")
    if (
        strided_row
        and count in (1024, 4095, 4096)
        and nfields == 2
        and nouts == 1
        and trait_key.startswith(("argmax", "argmin"))
    ):
        block = 32 if count == 1024 and dtype in (torch.float16, torch.bfloat16) else 64
        return _GeneralConfig(block=block, rule="b200_strided_index")
    if (
        strided_row
        and count in (1024, 4095, 4096)
        and nfields == 3
        and welford_nouts(trait_key) == nouts
    ):
        return _GeneralConfig(block=64, rule="b200_strided_welford", uniform_tree=True)
    if (
        strided_row
        and count in (1024, 4095, 4096)
        and nfields == 1
        and nouts == 1
        and trait_key in ("sum", "prod", "mean", "amax", "amin", "vnorm2")
    ):
        return _GeneralConfig(block=32, rule="b200_strided_one_field")
    if (
        cc != (10, 7)
        or dtype not in (torch.float32, torch.bfloat16)
        or count != 1024
        or tuple(red_pairs) != ((count, 2),)
        or tuple(kept_pairs) != ((num_o, 2 * count),)
        or acc_bits != 32
        or alignment < 16
    ):
        return _GeneralConfig()
    itemsize = 4 if dtype == torch.float32 else 2
    size = num_o * count * itemsize
    if size not in (16 << 20, 64 << 20, 256 << 20, 2 << 30):
        return _GeneralConfig()
    if size >= 256 << 20 and (trait_key, nfields, nouts) == ("sum", 1, 1):
        return _GeneralConfig(block=64, rule="rubin_strided_sum")
    if nfields == 3 and welford_nouts(trait_key) == nouts:
        return _GeneralConfig(kernel_order="inner_tree", rule="rubin_strided_welford")
    if size >= 64 << 20:
        if (
            nouts == 1
            and nfields == 1
            and trait_key in ("sum", "mean", "amax", "vnorm2")
        ):
            return _GeneralConfig(kernel_order="inner_tree", rule="rubin_strided_inner")
        if (trait_key, nfields, nouts) == ("argmaxi32", 2, 1) and (
            dtype == torch.bfloat16 or size == 2 << 30
        ):
            return _GeneralConfig(
                kernel_order="inner_tree", rule="rubin_strided_argmax"
            )
    return _GeneralConfig()


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
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
    tune: bool = True,
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

    M, itemsize = x.shape[0], x.element_size()
    order = rt.reduction_order(order)
    kernel_order = "inner_tree" if order == "inner_tree" else "linear"
    if tune:
        kernel_order = rt.select_row_order(
            _hw.caps(x.device).cc,
            x.dtype,
            trait_key,
            N,
            M,
            order=order,
            nfields=trait.nfields,
            nouts=nouts,
            acc_bits=max(dt.width for dt in trait.fdtypes),
            alignment=_L.supported_alignment(x, tile.TRANSFER_ALIGNMENT),
        )
    if kernel_order == "inner_tree":
        return rt.reduce_row_tile(
            trait, trait_key, x, out_dtypes, nouts=nouts, order="inner_tree"
        )
    ragged_vector = (
        tune
        and order == "unordered"
        and _hw.caps(x.device).cc == (10, 0)
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and tile.vec_size(N, itemsize) == 1
        and _L.supported_alignment(x, tile.TRANSFER_ALIGNMENT)
        == tile.TRANSFER_ALIGNMENT
        and (
            trait.nfields == 1
            or N >= 1024
            or x.dtype == torch.float32
            and trait.nfields == 2
        )
    )
    if ragged_vector:
        return rt.reduce_row_tile(
            trait,
            trait_key,
            x,
            out_dtypes,
            nouts=nouts,
            order="linear",
            ragged_vector=True,
        )
    if rt.one_thread_row_ok(N, itemsize, M, x.device):
        return rt.reduce_row_tile(
            trait,
            trait_key,
            x,
            out_dtypes,
            nouts=nouts,
            threads_per_row=1,
            order="linear",
        )
    if _oneshot_ok(x):
        return rt.reduce_row_tile(
            trait, trait_key, x, out_dtypes, nouts=nouts, order="linear"
        )

    from . import kernel_xcta as xc

    if nouts == 2:
        # The same fused split as nouts==1, projecting both fields. Without it a few-row/huge-N
        # 2-output reduction lands on one-block-per-row: 0.63x of ATen at N=65536, 0.20x at 131072.
        res = xc.reduce_row_xcta_2out(trait, trait_key, x, out_dtypes)
    else:
        res = xc.reduce_row_xcta(trait, trait_key, x, out_dtypes[0])
        res = None if res is None else (res,)
    if res is not None:
        return res
    # Prime sizes and index traits fall through to the ragged split.
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
    M, N = x.shape
    # A row split is the one-reduced-pair case of the general splitter.
    return _two_stage_general(
        trait,
        trait_key,
        x,
        [(N, 1)],
        kept_pairs=[(M, N)],
        num_o=M,
        count=N,
        out_dtypes=out_dtypes,
        nouts=nouts,
        block=block,
    )


def _two_stage_general(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    red_pairs: Pairs,
    kept_pairs: Pairs,
    num_o: int,
    count: int,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    block: int = _K0_ALL_BLOCK,
) -> tuple[torch.Tensor, ...] | None:
    # Split the slowest-varying reduced run to fill the device when kept coordinates do not.
    if not red_pairs:
        return None
    if getattr(trait, "has_index", False) and len(red_pairs) > 1:
        # A mixed-radix step index is not a reduced coordinate for multi-run index traits.
        return None
    sm = _hw.caps(x.device).sm_count
    E, S = red_pairs[-1]
    grid_mult = _K0_ALL_GRID_MULT
    if count * x.element_size() >= _WIDE_SPLIT_MIN_BYTES:
        grid_mult = getattr(trait, "split_grid_mult", grid_mult)
    want = -(-(sm * grid_mult) // max(1, num_o))
    # Total stage-1 CTAs stay near the architecture's occupancy target because
    # want scales inversely with num_o. A fixed per-row cap left B200 below one wave.
    C = max(1, min(want, E))
    if C == 1:
        return None
    e = -(-E // C)
    if S == 1:
        # Preserve transfer alignment for unit-stride chunks.
        vec = max(1, tile.TRANSFER_ALIGNMENT // x.element_size())
        e = min(E, max(vec, e // vec * vec))
    C = -(-E // e)
    if C == 1:
        return None
    inner = count // E
    parts = [
        torch.empty(num_o * C, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(num_o, device=x.device, dtype=d) for d in out_dtypes]

    # Put the chunk pair first so each kept coordinate's C partials are contiguous.
    s1 = ReduceBlock(
        trait,
        count=e * inner,
        num_o=num_o * C,
        red_pairs=list(red_pairs[:-1]) + [(e, S)],
        kept_pairs=[(C, e * S)] + list(kept_pairs),
        in_base=int(x.storage_offset()),
        limit=count,
        ragged_chunk=True,
        gidx_from="chunk" if getattr(trait, "has_index", False) else "r",
        nouts=trait.nfields,
        final=False,
        block=block,
    )
    _launch(s1, ("gensplit1", trait_key, x.dtype) + s1.cache_sig, [_flat(x)], parts)

    s2 = ReduceBlock(
        trait,
        count=C,
        num_o=num_o,
        red_pairs=[(C, 1)],
        kept_pairs=[(num_o, C)],
        from_partials=True,
        project_n=count,
        nouts=nouts,
        final=True,
        block=block,
    )
    part_dtypes = tuple(p.dtype for p in parts)
    key = ("gensplit2", trait_key, tuple(out_dtypes), part_dtypes) + s2.cache_sig
    _launch(s2, key, parts, _kernel_outs(outs))
    return tuple(outs)


def _indexed_itree_plan(
    trait: Any,
    count: int,
    num_o: int,
    itemsize: int,
    device: torch.device,
) -> "_ItreePlan | None":
    from . import kernel_rowtile as rt

    plan = rt.trait_itree_plan(
        trait, count, num_o, itemsize, stage=False, device=device
    )
    if plan is None:
        return None
    # General addressing cannot stage physically adjacent rows.
    return plan._replace(stage_rows=False)


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

    plan = _indexed_itree_plan(trait, count, num_o, x.element_size(), x.device)
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
    combine = rt.itree_combine_plan(
        plan,
        parts[0].element_size(),
        x.device,
        nfields=trait.nfields,
        nrows=num_o,
    )
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
    block: int | None = None,
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
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
    complex_input = getattr(trait, "complex_input", False)

    from . import kernel_rowtile as rt

    order = rt.reduction_order(order)
    tune = block is None
    block = _K0_BLOCK if block is None else block
    if order == "inner_tree":
        if complex_input:
            raise ValueError("inner-tree reduction cannot serve this geometry")
        if count >= _INT32_LIMIT or num_o >= _INT32_LIMIT:
            raise ValueError("inner-tree reduction requires 32-bit logical extents")
        kind = fast_kind(red_pairs, kept_pairs, nouts)
        if x.is_contiguous() and kind in ("row", "all"):
            x2 = x.reshape(num_o, count)
            ordered = rt.reduce_row_tile(
                trait,
                trait_key,
                x2,
                out_dtypes,
                nouts=nouts,
                order="inner_tree",
            )
        else:
            ordered = None
            from . import kernel_coltile as ct

            col_view = _physical_col_view(
                x, red_axes, red_pairs, kept_pairs, count, num_o
            )
            if col_view is not None:
                ordered = ct.reduce_ordered_col(
                    trait,
                    trait_key,
                    col_view,
                    out_dtypes,
                    nouts,
                    order=order,
                )
            if ordered is None:
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
        raise ValueError("inner-tree reduction cannot serve this geometry")

    # Route one-element outputs through the reduce-all split instead of one general block.
    if math.prod(out_shape) == 1 and x.is_contiguous():
        all_block = None if tune else block
        if nouts == 1:
            kwargs = {} if all_block is None else {"block": all_block}
            out = reduce_all(trait, trait_key, x, out_dtypes[0], order=order, **kwargs)
            return (_as_shape(out, out_shape),)
        kwargs = {} if all_block is None else {"block": all_block}
        outs = reduce_all2(trait, trait_key, x, out_dtypes, order=order, **kwargs)
        return tuple(_as_shape(o, out_shape) for o in outs)

    # Reshape dense TensorIterator row/column views onto fast kernels; general remains
    # the fallback for other layouts and declines.
    if (
        not complex_input
        and len(out_shape) > 0
        and count < _INT32_LIMIT
        and num_o < _INT32_LIMIT
    ):
        kind = fast_kind(red_pairs, kept_pairs, nouts)
        red_n = x.numel() // max(1, math.prod(out_shape))
        if kind == "row" and x.is_contiguous():
            x2 = x.reshape(math.prod(out_shape), red_n)
            fast = _try_fast_row(
                trait, trait_key, x2, out_dtypes, nouts, order=order, tune=tune
            )
            if fast is not None:
                return tuple(_as_shape(o, out_shape) for o in fast)
        col_view = _physical_col_view(x, red_axes, red_pairs, kept_pairs, red_n, num_o)
        if col_view is not None:
            from . import kernel_coltile as ct

            if col_view.dim() == 2:
                if nouts == 1:
                    fast = (
                        ct.reduce_col_tile(
                            trait, trait_key, col_view, out_dtypes[0], order=order
                        ),
                    )
                else:
                    fast = ct.reduce_col_tile_2out(
                        trait, trait_key, col_view, out_dtypes, order=order
                    )
            else:
                fast = ct.reduce_batched_col_tile(
                    trait, trait_key, col_view, out_dtypes, nouts, order=order
                )
            return tuple(_as_shape(out, out_shape) for out in fast)

    config = _GeneralConfig(block=block)
    if tune:
        config = select_general_config(
            _hw.caps(x.device).cc,
            x.dtype,
            trait_key,
            count,
            num_o,
            red_pairs,
            kept_pairs,
            order=order,
            nfields=trait.nfields,
            nouts=nouts,
            acc_bits=max(dt.width for dt in trait.fdtypes),
            alignment=_L.supported_alignment(x, tile.TRANSFER_ALIGNMENT),
        )
    if config.kernel_order == "inner_tree":
        ordered = _try_indexed_itree(
            trait, trait_key, x, red_pairs, kept_pairs, num_o, count, out_dtypes, nouts
        )
        if ordered is not None:
            return tuple(_as_shape(o, out_shape) for o in ordered)
    block = config.block

    # One block per kept coordinate is the whole grid, so split the reduced run whenever it
    # would not fill the SMs and there is enough work per output to pay for a second launch.
    sm = _hw.caps(x.device).sm_count
    if num_o < sm * _K0_ALL_GRID_MULT and count >= _SPLIT_MIN_COUNT:
        split = _two_stage_general(
            trait,
            trait_key,
            x,
            red_pairs,
            kept_pairs,
            num_o,
            count,
            out_dtypes,
            nouts,
            block,
        )
        if split is not None:
            return tuple(_as_shape(o, out_shape) for o in split)

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
        tree_count=count if config.uniform_tree else 0,
    )
    key = (
        "reduce",
        trait_key,
        x.dtype,
        tuple(out_dtypes),
        order,
        config,
    ) + op.cache_sig
    _launch(op, key, [_flat(x)], _kernel_outs(outs))
    return tuple(outs)


def reduce_dim(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    dims: int | Sequence[int] | None,
    out_dtype: torch.dtype,
    block: int | None = None,
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
) -> torch.Tensor:
    return _reduce(trait, trait_key, x, dims, [out_dtype], 1, block=block, order=order)[
        0
    ]


def reduce_dim2(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    dims: int | Sequence[int] | None,
    out_dtypes: Sequence[torch.dtype],
    block: int | None = None,
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
) -> tuple[torch.Tensor, ...]:
    return _reduce(
        trait, trait_key, x, dims, list(out_dtypes), 2, block=block, order=order
    )


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
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
) -> torch.Tensor:
    return _reduce_all(
        trait, trait_key, x, [out_dtype], 1, block, grid_mult, order=order
    )[0]


def reduce_all2(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    block: int = _K0_ALL_BLOCK,
    grid_mult: int = _K0_ALL_GRID_MULT,
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
) -> tuple[torch.Tensor, ...]:
    return _reduce_all(
        trait, trait_key, x, list(out_dtypes), 2, block, grid_mult, order=order
    )


def _reduce_all(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    block: int,
    grid_mult: int,
    *,
    order: Literal["unordered", "inner_tree"] | None = None,
) -> tuple[torch.Tensor, ...]:
    # Try the one-shot row kernel, fused cross-CTA split, then grid-striding fallback.
    # All preserve flat indices because reduce-all is a single row.
    if not x.is_cuda:
        raise AssertionError(f"reduce-all needs a CUDA input, got {x.device}")
    from . import kernel_rowtile as rt

    order = rt.reduction_order(order)
    if block == _K0_ALL_BLOCK and grid_mult == _K0_ALL_GRID_MULT:
        cfg = select_all_config(
            _hw.caps(x.device).cc,
            x.dtype,
            trait_key,
            x.numel(),
            nfields=trait.nfields,
            nouts=nouts,
        )
        block, grid_mult = cfg.block, cfg.grid_mult
    if not x.is_contiguous():
        return _reduce(
            trait, trait_key, x, None, out_dtypes, nouts, block=block, order=order
        )
    L = x.numel()
    complex_input = getattr(trait, "complex_input", False)
    xf = x.reshape(-1)
    if L == 1 and xf.stride(0) != 1:
        # A single element is contiguous at any stride; the wrap still requires unit stride.
        xf = xf.as_strided((1,), (1,))
    # Keep one-shot inputs out of xcta, which adds a ~1.9us launch or declines them.
    # Direct routing measured 1.2-2.1x over ATen.
    x2 = xf.view(1, -1)

    if order == "inner_tree":
        if complex_input:
            raise ValueError("inner-tree reduction cannot serve this geometry")
        if L >= _INT32_LIMIT:
            raise ValueError("inner-tree reduction requires 32-bit logical extents")
        outs = rt.reduce_row_tile(
            trait,
            trait_key,
            x2,
            out_dtypes,
            nouts=nouts,
            order="inner_tree",
        )
        return tuple(_as_shape(o, ()) for o in outs)
    if not complex_input and _oneshot_ok(x2):
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
            order="linear",
        )
        return tuple(_as_shape(o, ()) for o in outs)
    from . import kernel_xcta as xc

    if not complex_input:
        if nouts == 1:
            res = xc.reduce_row_xcta(trait, trait_key, xf, out_dtypes[0], flatten=True)
            res = None if res is None else (res,)
        else:
            res = xc.reduce_row_xcta_2out(
                trait, trait_key, xf, out_dtypes, flatten=True
            )
        if res is not None:
            return res
    # If xcta declines, grid-stride any L without reshaping or extent-dependent compile cost.
    sm = _hw.caps(x.device).sm_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(1, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1 models G contiguous chunks as kept (G, chunk), reduced (chunk, 1);
    # flat_tail guards the last and chunk indices stay relative to the input view.
    in_base = int(xf.storage_offset())
    s1 = ReduceBlock(
        trait,
        count=chunk,
        num_o=G,
        red_pairs=[(chunk, 1)],
        kept_pairs=[(G, chunk)],
        in_base=in_base,
        limit=in_base + L,
        flat_tail=True,
        gidx_from="chunk",
        nouts=trait.nfields,
        final=False,
        block=block,
    )
    _launch(s1, ("all1", trait_key, x.dtype) + s1.cache_sig, [_flat(x)], parts)

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
    # Key on the PARTIAL dtypes too: for an index trait out_dtypes is the index dtype and does
    # not track the value accumulator, so fp32 and fp64 argmax would share one cached kernel.
    part_dtypes = tuple(p.dtype for p in parts)
    s2_key = ("all2", trait_key, tuple(out_dtypes), part_dtypes) + s2.cache_sig
    _launch(s2, s2_key, parts, _kernel_outs(outs))
    return tuple(_as_shape(o, ()) for o in outs)
