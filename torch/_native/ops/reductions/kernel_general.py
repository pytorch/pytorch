# The general CuteDSL reduction kernel and the dispatcher for the whole taxonomy. One
# @cute.kernel handles ANY geometry -- row, column, n-D, transposed, sliced, reduce-all --
# through a TensorIterator-derived offset decode, which makes it the correctness floor and the
# COMBINE engine for the two-stage drivers that follow. Specialized kernels wire their own
# branch into `_reduce()` as each is introduced.
#
# ADDRESSING. TI coalesces any reduction into an iteration where a dim is REDUCED iff the
# output stride along it is 0. The host passes compile-time (extent, element_stride) lists and
# the kernel decodes a linear index to a flat offset against them, so multi-dim reductions and
# arbitrary strides need no special cases. Reduce-all is the zero-kept-dims case.
#
# Only STRUCTURE is compiled in; geometry VALUES are runtime launch args, so the kernel count
# is O(op x dtype x pair-count) rather than O(distinct shapes).

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, WARP, warp_reduce


def _magic(d):
    # Magic-number reciprocal for exact n // d as (n * m) >> sh: one multiply-shift instead of a
    # runtime 64-bit divide per element per pair. Granlund-Montgomery, as in aten's IntDivider,
    # in the round-up form the Int32-positive domain allows -- exact, and an instruction cheaper.
    l = (d - 1).bit_length()
    return (1 << (31 + l)) // d + 1, 31 + l


def _decode_offset(linear, vals, npairs):
    # Mixed-radix decode of a linear index to a flat element offset. `vals` is a RUNTIME quad list,
    # fastest dim first, so only the pair COUNT is baked and one kernel serves every geometry
    # sharing it; the last pair needs neither div nor mod. INT64 throughout, since numel can exceed
    # 2**31 and an int32 product wraps negative and reads out of bounds.
    rem = cutlass.Int64(linear)
    if npairs == 1:
        return rem * vals[3]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        q = (rem * vals[4 * j]) >> vals[4 * j + 1]
        off = off + (rem - q * vals[4 * j + 2]) * vals[4 * j + 3]
        rem = q
    return off + rem * vals[4 * (npairs - 1) + 3]


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
        from_partials=False,
        block=128,
    ):
        self.trait = trait
        self.count = count  # elements reduced per output (= prod red exts)
        self.num_o = num_o  # number of outputs / blocks (= prod kept exts)
        # The magic-division decode (_magic) is exact only for linear indices
        # < 2^31; r spans count and o spans num_o, both Int32 in the kernel.
        if not (count < 2**31 and num_o < 2**31):
            raise AssertionError(
                f"decode needs count and num_o < 2^31, got {count} and {num_o}"
            )
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
        self.gidx_from = gidx_from  # "r" (per-axis index) or "flat" (reduce-all)
        self.flat_tail = flat_tail  # clamp the fold bound to limit (reduce-all s1)
        self.from_partials = from_partials
        self.block = block
        self.num_warps = block // WARP

    @property
    def cache_sig(self):
        # STRUCTURE only: the counts, pair values and bounds are RUNTIME launch args, so one compiled
        # kernel serves every geometry sharing this structure. Callers prepend the trait key and dtypes.
        return (
            self.npairs_red,
            self.npairs_kept,
            self.nouts,
            self.final,
            self.gidx_from,
            self.flat_tail,
            self.from_partials,
            self.block,
            self.trait.nfields,
        )

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

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOuts: list,
        rvals: list,
        kvals: list,
        count: cutlass.Int32,
        in_base: cutlass.Int64,
        limit: cutlass.Int64,
        project_n: cutlass.Int64,
        stream,
    ):
        # Dynamic grid: read the output row count live so one compile serves any M.
        self.kernel(mIns, mOuts, rvals, kvals, count, in_base, limit, project_n).launch(
            grid=[mOuts[0].shape[0], 1, 1], block=[self.block, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mIns: list,
        mOuts: list,
        rvals: list,
        kvals: list,
        count: cutlass.Int32,
        in_base: cutlass.Int64,
        limit: cutlass.Int64,
        project_n: cutlass.Int64,
    ):
        trait = self.trait
        tidx, _, _ = cute.arch.thread_idx()
        o, _, _ = cute.arch.block_idx()
        nfields = const_expr(trait.nfields)

        acc = trait.init()
        # Base flat input offset for this block's KEPT coordinate (decode o
        # against the kept dims). 0 kept pairs (reduce-all) -> just in_base.
        obase = in_base
        if const_expr(self.npairs_kept > 0):
            obase = in_base + _decode_offset(o, kvals, self.npairs_kept)
        # Per-block fold bound: normally the full count, but with flat_tail clamped to the elements
        # left before `limit`, so the overhanging last chunk folds nothing out of range.
        rb = count
        if const_expr(self.flat_tail):
            left = limit - obase
            c64 = cutlass.Int64(count)
            left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
            zero = cutlass.Int64(0)
            left = left if left > zero else zero  # noqa: FURB136 -- no DSL builtin max
            rb = cutlass.Int32(left)
        n_full = rb // const_expr(self.block)
        reduce_fn = trait.reduce  # local bind: attribute access trips a dyn loop
        acc_dtype = trait.acc  # accumulator dtype (a compile-time Python class)
        if const_expr(self.from_partials):
            # Stage 2: COMBINE pre-reduced accumulator tuples from the per-field partial buffers. Each
            # output's partials are a contiguous run, and the base must be decoded from kept_pairs or every
            # row reads row 0's. The fold is a DYNAMIC full-wave loop plus a constexpr remainder, because a
            # static unroll scaled compile time with the partial count (~3s at 1e5). Bind the trait's
            # attributes to locals -- attribute access inside a dynamic loop trips the IR flattener.
            combine_fn = trait.combine
            fdtypes = trait.fdtypes
            nf = const_expr(nfields)
            r = tidx
            for _ in cutlass.range(n_full):
                rr = obase + cutlass.Int64(r)
                part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
                acc = combine_fn(acc, part)
                r = r + const_expr(self.block)
            # count is a runtime value now, so the remainder pass is always emitted
            # (predicated; a full-wave count just predicates every lane off).
            valid = r < rb
            rr = (obase + cutlass.Int64(r)) if valid else in_base
            part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
            merged = combine_fn(acc, part)
            acc = tuple((merged[f] if valid else acc[f]) for f in range(nf))
        else:
            # Per-axis fold. rb is pre-clamped, so `r < rb` already implies in-range and the per-element
            # guard collapses into it. The trip count is DYNAMIC and every full wave is all-in-range, so
            # the loop guard is a python constant and compile depth is O(1) in the extent.
            base_r = tidx
            for _ in cutlass.range(n_full):
                # Inline the offset (no intermediate name that the DSL would treat
                # as loop-carried across iterations). acc and base_r are the only
                # carried values; both are initialized before the loop. The "flat"
                # gidx recomputes the decode (single-pair there, so it is one mul).
                if const_expr(self.gidx_from == "flat"):
                    acc = reduce_fn(
                        acc,
                        acc_dtype(
                            mIns[0][
                                obase + _decode_offset(base_r, rvals, self.npairs_red)
                            ]
                        ),
                        Int32(obase + _decode_offset(base_r, rvals, self.npairs_red)),
                        True,
                    )
                else:
                    acc = reduce_fn(
                        acc,
                        acc_dtype(
                            mIns[0][
                                obase + _decode_offset(base_r, rvals, self.npairs_red)
                            ]
                        ),
                        base_r,
                        True,
                    )
                base_r = base_r + const_expr(self.block)
            # Invalid lanes read in_base (always in range) -- obase itself can be
            # past the end for an overhanging reduce-all chunk (rb clamped to 0).
            valid = base_r < rb
            off = obase + _decode_offset(base_r, rvals, self.npairs_red)
            off_s = off if valid else in_base
            val = acc_dtype(mIns[0][off_s])
            # gidx is the argmax index fed to the trait (Int32 domain): "flat" =
            # the global flat input offset (reduce-all; fits int32 per-chunk).
            if const_expr(self.gidx_from == "flat"):
                acc = reduce_fn(acc, val, Int32(off_s), valid)
            else:
                acc = reduce_fn(acc, val, base_r, valid)

        acc = warp_reduce(trait, acc, WARP)
        if const_expr(self.num_warps > 1):
            smem = cutlass.utils.SmemAllocator()
            bufs = [
                smem.allocate_tensor(
                    trait.fdtypes[f], cute.make_layout(self.num_warps), byte_alignment=8
                )
                for f in range(nfields)
            ]
            acc = block_reduce(trait, acc, bufs, self.num_warps)

        if const_expr(self.final):
            # project (post-op) applied exactly once; store nouts result(s).
            # project_n is the TRUE reduction size (= count single-stage; = L for
            # reduce-all stage 2, where count is just the partial count G). A
            # runtime value: the Int64 -> acc-dtype convert happens in-kernel.
            result = trait.project(acc, acc_dtype(project_n))
            if tidx == 0:
                if const_expr(self.nouts == 1):
                    mOuts[0][o] = mOuts[0].element_type(result)
                else:
                    for k in cutlass.range_constexpr(self.nouts):
                        mOuts[k][o] = mOuts[k].element_type(result[k])
        else:
            # Cross-CTA stage 1: store the RAW (pre-project) accumulator fields.
            if tidx == 0:
                for f in cutlass.range_constexpr(nfields):
                    mOuts[f][o] = trait.fdtypes[f](acc[f])


# Host plumbing + the geometry chooser: these build the plans that drive tile.TileReduce.
# Nothing here is a kernel.
_stream = _L.stream
# _L.compile_kernel: cute.compile against FAKE operands + options="--enable-tvm-ffi", so the
# compiled callable takes the torch tensors and there is no per-call wrap.
_compile = _L.compile_kernel
_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}

_COMPILE_CACHE = {}  # structural key -> compiled kernel (one per cache_sig)
_PLAN = {}  # (structural key, geom_sig) -> (compiled fn, pre-boxed geometry args)


def _fakes(ts):
    # Compile-time descriptors. Every operand is a 1D flat view whose extent is DYNAMIC, so one
    # structural kernel serves any length -- required, since the grid reads a shape live.
    return [_L.fake_compact(torch2cute[t.dtype], (_L.sym(),)) for t in ts]


def _operands(ts, read_only=False):
    # The real tensors, as the compiled callable takes them. INPUTS go through read_only(), or a COW
    # input is materialized on export.
    return [_L.read_only(t) for t in ts] if read_only else list(ts)


def _quads(pairs):
    # (extent, stride) pairs -> the flat quad list _decode_offset consumes. Runs once per NEW
    # geometry (the boxed result is memoized), so the divide cost is off the repeat path.
    out = []
    for ext, strd in pairs:
        m, sh = _magic(ext)
        out += [Int64(m), Int64(sh), Int64(ext), Int64(strd)]
    return out


def _geom_args(op):
    # The RUNTIME geometry of a launch: magic-division quads for the two decodes plus the scalar
    # bounds, all of which used to be baked const_exprs. The magic form needs indices < 2**31,
    # which count/num_o assert. Unused row/col args are None, not dummies: an unused Int32 kernel
    # parameter is not free (see tile.TileReduce.kernel).
    return (
        _quads(op.red_pairs),
        _quads(op.kept_pairs),
        Int32(op.count),
        Int64(op.in_base),
        Int64(op.limit),
        Int64(op.project_n),
    )


def _launch(op, key, ins, outs):
    # Two-level cache: _PLAN memoizes the pre-boxed launch args per GEOMETRY, since boxing ten
    # scalars costs ~6us; _COMPILE_CACHE dedupes the compile per STRUCTURE.
    plan = _PLAN.get((key, op.geom_sig))
    if plan is None:
        fn = cached_plan(
            _COMPILE_CACHE,
            key,
            lambda: _compile(op, _fakes(ins), _fakes(outs), *_geom_args(op), _stream()),
            op=f"aten::{key[1]}",
        )
        plan = (fn, _geom_args(op))
        _PLAN[(key, op.geom_sig)] = plan
    fn, geom = plan
    fn(_operands(ins, read_only=True), _operands(outs), *geom, _stream())


def _ti_pairs(x, out):
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


def _probe(x, red_axes):
    # A dummy output with the reduced dims set to 1, as reduce_op expects (it reads shapes and
    # strides only). Shared by the classifier and the fallback so both see one TI decode.
    return torch.empty(
        [1 if i in red_axes else s for i, s in enumerate(x.shape)],
        device=x.device,
        dtype=x.dtype,
    )


def _flat(x):
    # A 1D stride-1 view over x's ENTIRE storage: TI's element strides are storage-relative, and
    # x.reshape(-1) on a non-contiguous x would copy and break the stride math.
    n = max(x.untyped_storage().nbytes() // x.element_size(), 1)
    return torch.as_strided(x, (n,), (1,), storage_offset=0)


# --- Fast-path classification, the ONE source of truth for the router and the cond gate. It
# runs on the TI-decomposed pairs, so it sees POST-coalesce geometry. ---


def fast_kind(red_pairs, kept_pairs, nouts, has_index):
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
    if kept_pairs[0][1] == 1 and nouts == 1 and not has_index:  # kept innermost -> col
        return "col"
    return None


# The general axis's launch config as named DATA. It is the any-geometry backstop rather than
# a perf path, so these are occupancy baselines and not a tuned surface.
_K0_BLOCK = 128
_K0_ALL_BLOCK = 256
_K0_ALL_GRID_MULT = 4


def _as_shape(out, out_shape):
    # Give the flat output its n-D shape WITHOUT leaving it a view: the kernels allocate their own
    # buffer, and an aten reduction never aliases -- OpInfo's python-ref tests check that.
    if tuple(out.shape) == tuple(out_shape):
        return out
    reshaped = out.reshape(out_shape)
    if reshaped._base is None:
        return reshaped
    out.resize_(out_shape)
    return out


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


def _grid_size(L, block, sm_count, grid_mult=4):
    # G = stage-1 chunks. Fill the device to grid_mult waves, capped by the work available: more
    # chunks means more stage-1 parallelism but a larger stage-2 fold.
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(
    trait, trait_key, x, out_dtype, block=_K0_ALL_BLOCK, grid_mult=_K0_ALL_GRID_MULT
):
    # Full-tensor reduce-all via the two-stage split: stage 1 grid-strides the flat input into G
    # chunks, stage 2 folds them and projects once. Mirrors ATen's ctas_per_output and needs no
    # reshape, so it serves any element count. Index traits are served too, since a single row
    # makes the flat offset the column.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"reduce-all needs a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    L = x.numel()
    xf = x.reshape(-1)
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    out = torch.empty(1, device=x.device, dtype=out_dtype)

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
        nouts=1,
        final=True,
        block=block,
    )
    _launch(s2, ("all2", trait_key, out_dtype) + s2.cache_sig, parts, [out])
    return out.reshape(())
