# The reduction DISPATCHER, plus the plan for the shared kernel's GENERAL axis. Every
# reduction enters `_reduce()`, which decodes the geometry with TensorIterator and routes to
# one of tile.TileReduce's axes -- the only @cute.kernel in the family. The general axis
# addresses through a TI offset decode, so no layout is declined for its own sake.
#
# Only STRUCTURE is compiled in (cache_sig); geometry VALUES are runtime args, so the kernel
# count is O(op x dtype x pair-count), not O(distinct shapes). The const_expr keys are
# npairs_red/npairs_kept (decode depth), nouts, gidx_from (what an index trait is told the
# position is), final, from_partials, and the two fold-bound clamps.

import math

from cutlass import Float32, Float64, Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import WARP
from . import tile


# (extent, element-stride) pairs from TensorIterator, fastest dim first.
Pairs = list[tuple[int, int]]


class ReduceBlock:
    def __init__(
        self,
        trait,
        *,
        count,
        num_o,
        red_pairs,
        kept_pairs,
        in_base=0,
        limit=None,
        project_n=None,
        nouts=1,
        final=True,
        gidx_from="r",
        flat_tail=False,
        ragged_chunk=False,
        from_partials=False,
        block=128,
    ):
        self.trait = trait
        self.count = count  # elements reduced per output (= prod red exts)
        self.num_o = num_o  # number of outputs / blocks (= prod kept exts)
        # The decode runs in Int32: r spans count and o spans num_o.
        if not (count < 2**31 and num_o < 2**31):
            raise AssertionError(
                f"decode needs count and num_o < 2^31, got {count} and {num_o}"
            )
        if not red_pairs:
            # A plan with no REDUCED runs would index vals[-1] in the fold's decode. An empty KEPT list
            # is legal (a full reduction): the body drops the kept decode for it.
            raise AssertionError("a reduction needs at least one reduced run")
        # (extent, input-element-stride) pairs from TensorIterator, fastest first.
        self.red_pairs = tuple(red_pairs)
        self.kept_pairs = tuple(kept_pairs)
        self.npairs_red = len(self.red_pairs)
        self.npairs_kept = len(self.kept_pairs)
        self.in_base = in_base  # flat input offset of output coordinate 0
        self.limit = limit if limit is not None else count  # ragged tail bound
        # The N passed to project (mean's divisor). = count single-stage; = L for
        # reduce-all stage 2 (where count is the partial count G, not L).
        self.project_n = project_n if project_n is not None else count
        self.nouts = nouts
        self.final = final
        self.gidx_from = gidx_from  # "r" | "flat" (reduce-all) | "chunk" (row split)
        self.flat_tail = flat_tail  # clamp the fold bound to limit (reduce-all s1)
        self.ragged_chunk = ragged_chunk  # clamp to this output's reduced run
        self.from_partials = from_partials
        self.block = block
        # The BODY is the shared kernel's general axis: one block per output, every thread folding
        # that output through the mixed-radix decode. This class is the plan and owns no kernel.
        self.tile = tile.TileReduce(
            trait,
            None,
            "general",
            0,
            nt=block,
            nouts=nouts,
            final=final,
            combine=from_partials,
            npairs_red=self.npairs_red,
            npairs_kept=self.npairs_kept,
            gidx_from=gidx_from,
            flat_tail=flat_tail,
            ragged_chunk=ragged_chunk,
        )

    @property
    def cache_sig(self):
        # DERIVED from the body, not restated: the body bakes the const_exprs, so a knob added there
        # would be invisible to this key and a stale kernel silently reused.
        return self.tile.cache_sig

    @property
    def geom_sig(self):
        # The RUNTIME geometry, for the per-geometry PLAN cache: a repeat geometry skips the
        # Int32/Int64 boxing (~6us/launch) but still shares the structurally-cached kernel.
        return (
            self.count,
            self.red_pairs,
            self.kept_pairs,
            self.in_base,
            self.limit,
            self.project_n,
        )


# Host plumbing + the geometry chooser: these build the plans that drive tile.TileReduce.
# Nothing here is a kernel.
_stream = _L.stream
# _L.compile_kernel: cute.compile against FAKE operands + options="--enable-tvm-ffi", so the
# compiled callable takes the torch tensors and there is no per-call wrap.
_compile = _L.compile_kernel
_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}

_COMPILE_CACHE = {}  # structural key -> compiled kernel (one per cache_sig)
_PLAN = {}  # (structural key, geom_sig) -> (compiled fn, pre-boxed geometry args)


def _fakes(ts: list[torch.Tensor]) -> list:
    # Compile-time descriptors. Every operand is a 1D flat view whose extent is DYNAMIC, so one
    # structural kernel serves any length -- required, since the grid reads a shape live.
    return [_L.fake_compact(torch2cute[t.dtype], (_L.sym(),)) for t in ts]


def _operands(ts: list[torch.Tensor], read_only: bool = False) -> list:
    # The real tensors, as the compiled callable takes them. INPUTS go through read_only(), or a COW
    # input is materialized on export.
    return [_L.read_only(t) for t in ts] if read_only else list(ts)


def _exts(pairs: Pairs) -> list:
    # Extents for the decode's divisors, which the launch turns into FastDivmod objects inside
    # the traced region (they need an MLIR context, so they cannot be built here).
    return [Int32(ext) for ext, _ in pairs]


def _strides(pairs: Pairs) -> list:
    return [Int64(strd) for _, strd in pairs]


def _geom_args(op):
    # The RUNTIME geometry of a launch: the two decodes' extents and strides plus the scalar
    # bounds, all of which used to be baked const_exprs.
    return (
        Int32(op.count),
        None,
        Int64(op.project_n),
        None,
        None,
        _exts(op.red_pairs),
        _strides(op.red_pairs),
        _exts(op.kept_pairs),
        _strides(op.kept_pairs),
        Int64(op.in_base),
        Int64(op.limit),
    )


def _launch(op, key, ins, outs):
    # Two-level cache: _PLAN memoizes the pre-boxed launch args per GEOMETRY, since boxing ten
    # scalars costs ~6us; _COMPILE_CACHE dedupes the compile per STRUCTURE.
    plan = _PLAN.get((key, op.geom_sig))
    if plan is None:
        fn = cached_plan(
            _COMPILE_CACHE,
            key,
            lambda: _compile(
                op.tile, _fakes(ins), _fakes(outs), *_geom_args(op), _stream()
            ),
            op=f"aten::{key[1]}",
        )
        plan = (fn, _geom_args(op))
        _PLAN[(key, op.geom_sig)] = plan
    fn, geom = plan
    fn(_operands(ins, read_only=True), _operands(outs), *geom, _stream())


def _ti_pairs(x: torch.Tensor, out: torch.Tensor) -> tuple[Pairs, Pairs]:
    """Input addressing for ``reduce x into out``, off TensorIterator: a dim is REDUCED iff the
    output stride along it is 0. Returns (red_pairs, kept_pairs) of (extent, input stride).

    KEPT dims are ordered by OUTPUT stride ascending, because the block index is decoded
    fastest-first. REDUCED dims need no order -- the fold visits each element once.
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
    # A dummy output with the reduced dims set to 1, as reduce_op expects (it reads shapes and
    # strides only). Shared by the classifier and the fallback so both see one TI decode. It must
    # be its OWN allocation, not a view of x: TI takes an output's writable pointer, which would
    # materialize a COW input.
    return torch.empty(
        [1 if i in red_axes else s for i, s in enumerate(x.shape)],
        device=x.device,
        dtype=x.dtype,
    )


def _flat(x: torch.Tensor) -> torch.Tensor:
    # A 1D stride-1 view over x's ENTIRE storage: TI's element strides are storage-relative, and
    # x.reshape(-1) on a non-contiguous x would copy and break the stride math.
    n = max(x.untyped_storage().nbytes() // x.element_size(), 1)
    return torch.as_strided(x, (n,), (1,), storage_offset=0)


# --- Fast-path classification, the ONE source of truth for the router and the cond gate. It
# runs on the TI-decomposed pairs, so it sees POST-coalesce geometry. ---


def fast_kind(red_pairs: Pairs, kept_pairs: Pairs, nouts: int) -> str | None:
    """Which fast kernel serves this TI-decomposed reduction, or None for the general one.

    BOTH axes must coalesce to a single run, so the reduction is a dense 2D view; the stride-1
    pair is the innermost axis and decides row against col. "col" is value-only.
    """
    if len(kept_pairs) == 0:
        return "all"
    if len(red_pairs) != 1 or len(kept_pairs) != 1:
        return None
    if red_pairs[0][1] == 1:  # reduced run is innermost/contiguous -> row
        return "row"
    if kept_pairs[0][1] == 1 and nouts == 1:  # kept innermost -> col
        return "col"
    return None


# The one-shot stages a whole row tile (~N*itemsize), so it must fit smem; above that the
# multi-CTA split caps each chunk's tile. Conservative, to leave the reduction buffer room.
_SMEM_BUDGET = 192 * 1024
# ... and the per-thread LOAD count must stay bounded. It only runs away when the vector
# width collapses to 1 (an odd or prime N): measured 0.08-0.17x of ATen with no bound, and
# 1.93-2.41x once the cross-CTA split serves those instead. 64 separates every measured good
# case from every bad one. tile.MAX_UNROLL bounds the same quantity inside the kernel.
_ONESHOT_MAX_LOADS = 64
# Chunks per row for the ragged split (_two_stage_row). Caps the stage-2 fold.
_C_MAX_ROW = 64

# The general axis's launch config as named DATA. It is the any-geometry backstop rather than
# a perf path, so these are occupancy baselines and not a tuned surface.
_K0_BLOCK = 128
_K0_ALL_BLOCK = 256
_K0_ALL_GRID_MULT = 4


def _oneshot_ok(x):
    # One-shot: does the row fit its tile (~N elements of the input dtype) AND stay inside
    # the per-thread load bound?
    N = x.shape[-1]
    if N * x.element_size() > _SMEM_BUDGET:
        return False
    from . import kernel_rowtile as rt

    width = x.element_size() * 8
    vec = math.gcd(N, 128 // width)
    tpr = max(WARP, rt.row_config(N, width).tpr)
    return -(-N // (tpr * vec)) <= _ONESHOT_MAX_LOADS


def _try_fast_row(trait, trait_key, x, out_dtypes, nouts):
    # Fast path for the CONTIGUOUS last dim of a 2D problem; None if it is not handled. The
    # one-shot needs no index remap, so it serves index traits directly, while the cross-CTA
    # split declines them -- its reshape makes a sub-row's chunk index row % C, which is awkward
    # to rebase to a global column (see kernel_xcta's has_index gate).
    if x.dim() != 2 or x.stride(-1) != 1:
        return None
    N = x.shape[-1]
    if N < 1:
        return None
    if nouts not in (1, 2):
        return None
    from . import kernel_rowtile as rt

    # NARROW rows first: packed onto a row, threads_per_row floors at one WARP, so a row narrower
    # than WARP vec-chunks leaves most of each warp idle (25% at N=32). tpr=1 needs no merge.
    #
    # NOT when the reproducible order is on: tpr is a LAUNCH-SHAPE preference and the order
    # supersedes those, deriving its own thread map from N. Passing tpr made the fold decline the
    # order, so a narrow row returned the default order's bits while the gate claimed ATen's --
    # MEASURED as differing bits at (524288, 16) and (524288, 128) through the aten entry point,
    # and invisible to the golden-hash test, which calls the fold directly.
    if rt.narrow_row(N, x.element_size(), x.shape[0]):
        return rt.reduce_row_tile(trait, trait_key, x, out_dtypes, nouts=nouts, tpr=1)
    if _oneshot_ok(x):
        return rt.reduce_row_tile(trait, trait_key, x, out_dtypes, nouts=nouts)
    from . import kernel_xcta as xc

    if nouts == 2:
        # The same fused split as nouts==1, projecting both fields. Without it a few-row/huge-N
        # 2-output reduction lands on one-block-per-row: 0.63x of ATen at N=65536, 0.20x at 131072.
        res = xc.reduce_row_xcta_2out(trait, trait_key, x, out_dtypes)
        if res is not None:
            return res
        # xcta declined (no divisor split for this N) -> ragged split, same as nouts==1.
        return _two_stage_row(trait, trait_key, x, out_dtypes, nouts)
    res = xc.reduce_row_xcta(trait, trait_key, x, out_dtypes[0])
    if res is not None:
        return (res,)
    # xcta declined (a prime N, or an index trait): split raggedly instead -- same two stages, but
    # the chunk need not divide the row and stage 1 can carry the absolute column.
    return _two_stage_row(trait, trait_key, x, out_dtypes, nouts)


def _as_shape(out: torch.Tensor, out_shape: list[int]) -> torch.Tensor:
    # Give the flat output its n-D shape WITHOUT leaving it a view: the kernels allocate their own
    # buffer, and an aten reduction never aliases -- OpInfo's python-ref tests check that.
    if tuple(out.shape) == tuple(out_shape):
        return out
    reshaped = out.reshape(out_shape)
    if reshaped._base is None:
        return reshaped
    out.resize_(out_shape)
    return out


def _two_stage_row(trait, trait_key, x, out_dtypes, nouts, block=_K0_ALL_BLOCK):
    # RAGGED cross-CTA row split, for an N with no divisor in xcta's window: without it a prime N
    # lands on one block per row, measured 0.28x of ATen at (8, 131071). The chunk need not divide
    # N, so stage 1 clamps its fold to the end of the row. None at C == 1, which buys nothing.
    #
    # Index traits ARE served here, which is what lets xcta decline them: stage 1 sees the GLOBAL
    # column, so stage 2 needs no remap and ATen's first-wins tie-break survives.
    M, N = x.shape
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    # Enough chunks to fill the device, then round s up to a 16B-friendly multiple so the
    # per-chunk base stays aligned; C follows from s, and the tail is whatever is left.
    C = max(1, min(_C_MAX_ROW, -(-(sm * _K0_ALL_GRID_MULT) // max(1, M))))
    if C == 1:
        return None
    vec = max(1, 16 // x.element_size())
    s_chunk = max(vec, -(-N // C) // vec * vec)
    C = -(-N // s_chunk)
    if C == 1:
        return None

    parts = [
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1: one output per (row, chunk). The chunk pair is FASTEST-varying, which is what lets
    # the ragged clamp take its chunk index from the front of the kept lists.
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

    # Stage 2: fold the C partials of each row, project once with the TRUE row length.
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


def _reduce(trait, trait_key, x, dims, out_dtypes, nouts, block=_K0_BLOCK):
    # General reduction of x over `dims` (int / tuple / None), driven by TI: row, column, n-D,
    # transposed and sliced all take the same path. Returns nouts tensors.
    if not x.is_cuda:
        raise AssertionError(f"need a CUDA input, got {x.device}")
    red_axes = (
        range(x.dim()) if dims is None else ([dims] if isinstance(dims, int) else dims)
    )
    red_axes = {d % x.dim() for d in red_axes}
    out_shape = [s for i, s in enumerate(x.shape) if i not in red_axes]

    # A single output ELEMENT takes reduce_all's two-stage split, not the general fallback, which
    # would put ONE block on the whole row. Reached by a full `dims` set and by the M=1 row case,
    # where TI collapses the extent-1 kept axes away before the classifier can see them.
    if math.prod(out_shape) == 1 and nouts == 1 and x.is_contiguous():
        out = reduce_all(trait, trait_key, x, out_dtypes[0], block=block)
        return (_as_shape(out, out_shape),)

    # Classify the POST-TI-coalesce geometry and reshape onto a fast kernel, which is what puts a
    # contiguous n-D reduction over its innermost axes on the row/col path. The general kernel
    # stays the correctness fallback for direct callers and for a fast kernel that declines.
    if len(out_shape) > 0 and x.is_contiguous():
        red_pairs, kept_pairs = _ti_pairs(x, _probe(x, red_axes))
        kind = fast_kind(red_pairs, kept_pairs, nouts)
        red_n = x.numel() // max(1, math.prod(out_shape))
        if kind == "row":
            x2 = x.reshape(math.prod(out_shape), red_n)
            fast = _try_fast_row(trait, trait_key, x2, out_dtypes, nouts)
            if fast is not None:
                return tuple(o.reshape(out_shape) for o in fast)
        elif kind == "col":
            # The tile body's COLUMN axis splits the REDUCED axis, so the reduction carries parallelism of
            # its own instead of relying on the column count -- which is what makes a tall-narrow input
            # work: 7.24x of ATen at (65536, 256), 2.53x at (16384, 1024), 1.49x at (4096, 4096).
            from . import kernel_coltile as ct

            x2 = x.reshape(red_n, math.prod(out_shape))
            out = ct.reduce_col_tile(trait, trait_key, x2, out_dtypes[0])
            return (_as_shape(out, out_shape),)

    outs = [torch.empty(out_shape, device=x.device, dtype=d) for d in out_dtypes]
    num_o = max(1, math.prod(out_shape))  # blocks (kept coordinates)
    count = x.numel() // num_o  # elements reduced per output
    red_pairs, kept_pairs = _ti_pairs(x, _probe(x, red_axes))
    op = ReduceBlock(
        trait,
        count=count,
        num_o=num_o,
        red_pairs=red_pairs,
        kept_pairs=kept_pairs,
        in_base=x.storage_offset(),
        nouts=nouts,
        block=block,
    )
    key = ("reduce", trait_key, x.dtype, tuple(out_dtypes)) + op.cache_sig
    _launch(op, key, [_flat(x)], [o.reshape(-1) for o in outs])
    return tuple(outs)


def reduce_dim(trait, trait_key, x, dims, out_dtype, block=_K0_BLOCK):
    return _reduce(trait, trait_key, x, dims, [out_dtype], 1, block=block)[0]


def reduce_dim2(trait, trait_key, x, dims, out_dtypes, block=_K0_BLOCK):
    return _reduce(trait, trait_key, x, dims, list(out_dtypes), 2, block=block)


def _grid_size(L: int, block: int, sm_count: int, grid_mult: int = 4) -> int:
    # G = stage-1 chunks. Fill the device to grid_mult waves, capped by the work available: more
    # chunks means more stage-1 parallelism but a larger stage-2 fold.
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(
    trait, trait_key, x, out_dtype, block=_K0_ALL_BLOCK, grid_mult=_K0_ALL_GRID_MULT
):
    return _reduce_all(trait, trait_key, x, [out_dtype], 1, block, grid_mult)[0]


def _reduce_all(trait, trait_key, x, out_dtypes, nouts, block, grid_mult):
    # Full-tensor reduce-all, in preference order: the one-shot row kernel, then the fused
    # cross-CTA two-stage, then the grid-striding general one. Index traits are served throughout,
    # since a single row makes each sub-row's global column the flat index.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"reduce-all needs a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    L = x.numel()
    xf = x.reshape(-1)
    # Fits the one-shot tile -> no cross-CTA split is wanted. Left to xcta such an input either
    # folds a SINGLE partial in a kernel of its own (~1.9us of pure launch) or is declined below
    # its sub-row floor -- one kernel too many either way. Measured 1.2-2.1x over ATen.
    x2 = xf.view(1, -1)
    if _oneshot_ok(x2):
        from . import kernel_rowtile as rt

        # The launch is ONE row, so the ladder's row-packing tpr would leave the device on a fraction
        # of one CTA. rt.single_row_config returns None when the ladder's pick already stands.
        cfg = rt.single_row_config(L, x.element_size() * 8)
        kw = {} if cfg is None else {"tpr": cfg.tpr, "nt": cfg.nt}
        outs = rt.reduce_row_tile(trait, trait_key, x2, out_dtypes, nouts=nouts, **kw)
        return tuple(_as_shape(o, ()) for o in outs)
    from . import kernel_xcta as xc

    if nouts == 1:
        res = xc.reduce_row_xcta(trait, trait_key, xf, out_dtypes[0], flatten=True)
        res = None if res is None else (res,)
    else:
        res = xc.reduce_row_xcta_2out(trait, trait_key, xf, out_dtypes, flatten=True)
    if res is not None:
        return res
    # Too big for the one-shot and xcta declined (a prime L): the two-stage general path
    # grid-strides any L with no reshape, so compile stays O(1) and the device still fills.
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(1, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1: the 1D input split into G contiguous chunks, modelled as kept (G, chunk) and
    # reduced (chunk, 1), with flat_tail guarding the last chunk. gidx_from="flat", so an index
    # trait carries the true global offset.
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

    # Stage 2: fold the G per-field partials in one block and project once, with the divisor the
    # TRUE element count rather than G. That divisor is in cache_sig, so no stale one is reused.
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
