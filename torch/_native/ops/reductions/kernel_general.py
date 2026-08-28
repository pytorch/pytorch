# General CuTeDSL reduction kernel (K0) + the dispatcher for the whole fold-
# reduction taxonomy. K0 is one @cute.kernel that handles ANY geometry (row,
# column, n-D, transposed, sliced, reduce-all) via a TensorIterator-derived
# offset decode -- the correctness floor, and the COMBINE engine for the two-stage
# drivers that follow (from_partials). `_reduce()` is the dispatcher every reduction
# enters: it decodes the geometry and routes to the specialized kernels, each of which
# wires its own branch in as it is introduced:
#   contiguous last-dim, smem-fits   -> kernel_rowtile   (one-shot)
#   contiguous last-dim, larger N    -> kernel_xcta      (fused cross-CTA split)
#   every kept extent 1              -> reduce_all
#   anything only K0 could serve     -> declined to aten by the cond
#
# The trait protocol (init/reduce/combine/shfl_down/project, nfields/fdtypes) and
# the warp_reduce/block_reduce helpers are REUSED UNCHANGED from reduce_traits.
#
# How one kernel covers everything -- block `o` (one per KEPT-dim coordinate)
# reduces `count` elements along the reduced dims; threads stride the linear
# reduction index r by blockDim; then warp + block reduce.
#
# ADDRESSING is driven by torch's TensorIterator (see kernel_general /
# _decompose_via_ti). TI coalesces/reorders ANY reduction (any dim set, any
# strides, n-D) into an iteration where a dim is REDUCED iff the output stride
# along it is 0, else KEPT. The host passes two compile-time lists of
# (extent, element_stride) pairs over the INPUT tensor:
#     red_dims : the reduced dims  -> the per-thread fold walks these
#     kept_dims: the kept dims     -> one block per kept coordinate
# The kernel turns a linear index into a flat input offset by mixed-radix
# decode against those pairs (decode_offset). For the common single-reduced-run
# case this collapses to base + r*rstride; multi-dim reductions (e.g. dim=(1,3))
# and arbitrary strides (transpose, slice) fall out of the same decode with no
# special cases -- replacing the old x.t()/reshape geometry hacks.
#
# Reduce-all is just the degenerate "all dims reduced, zero kept dims" case for
# stage 1, plus a flat stage-2 fold of the G partials.
#
# Only STRUCTURE is compiled in (cache_sig); the geometry VALUES are runtime
# launch args, so one compiled kernel serves every reduction sharing a structure
# (kernel count is O(op x dtype x pair-count), not O(distinct shapes/strides)).
#
# const_expr policy flags (compiled away -> specialized SASS, no runtime br):
#   npairs_red / npairs_kept : the (extent, stride) pair COUNTS (decode depth).
#   nouts        : 1 or 2 result tensors (var_mean / max.dim are 2).
#   gidx_from    : "r" -> argmax index is the linear reduction index r (per-axis
#                  reductions); "flat" -> the global flat input offset (reduce-all).
#   final        : True  -> project the accumulator, store nouts result(s);
#                  False -> store the RAW accumulator fields to per-field gmem
#                           partial buffers (cross-CTA stage 1).
#   from_partials: True  -> per-thread step COMBINES pre-reduced accumulator
#                  tuples read from nfields partial buffers (stage 2);
#                  False -> it REDUCEs raw scalar inputs.
#   flat_tail    : clamp the per-block fold bound to `limit` (reduce-all stage 1,
#                  where the last chunk overhangs the flat input).
#
# runtime geometry args (Int32/Int64 launch args, NOT in the compile key):
#   rvals/kvals  : interleaved (extent, stride) Int64 lists for reduced/kept dims.
#   count        : elements reduced per output; in_base: flat input offset of
#   output coordinate 0; limit: ragged bound; project_n: mean's true divisor.
#
# I/O passed as python lists (cute kernels accept lists):
#   mIns  : [mX]                         when from_partials is False
#           [p0, ... p(nfields-1)]       when from_partials is True (stage 2)
#   mOuts : [o0] or [o0, o1]             when final is True
#           [p0, ... p(nfields-1)]       when final is False (stage 1 partials)

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, WARP, warp_reduce


def _magic(d):
    # Magic-number reciprocal for exact n // d over 0 <= n < 2^31 as
    # (n * m) >> sh -- one 64-bit multiply + shift instead of a runtime 64-bit
    # divide (which the ~%25-slower runtime-geometry decode would otherwise emit
    # per element per pair). Same Granlund-Montgomery family as aten's IntDivider
    # (aten/src/ATen/cuda/detail/IntegerDivider.cuh, hackersdelight.org/magic.htm);
    # that one uses the add-indicator form for the full unsigned 2^32 domain, but
    # K0's linear indices are Int32-positive (< 2^31), so the simpler round-up form
    # is exact and one instruction cheaper. Proof sketch: m = floor(2^(31+l)/d)+1
    # with l = ceil(log2 d), so m*d = 2^(31+l) + e with 0 < e <= d, and for n < 2^31
    # the error term n*e/(d*2^(31+l)) < 2^-l * 1 < 1/d ... floor((n*m) >> (31+l))
    # = n//d exactly. m < 2^32 (d > 2^(l-1)) so n*m < 2^63: no Int64 overflow.
    l = (d - 1).bit_length()
    return (1 << (31 + l)) // d + 1, 31 + l


def _decode_offset(linear, vals, npairs):
    # Mixed-radix decode of a linear index into a flat element offset. vals is a
    # RUNTIME Int64 list of QUADS, fastest-varying dim first:
    #     [m0, sh0, ext0, strd0,  m1, sh1, ext1, strd1, ...]
    # where (m, sh) is _magic(ext). npairs is the compile-time pair COUNT (only the
    # loop STRUCTURE is baked -- the values are launch args, so one compiled kernel
    # serves every geometry with the same pair count). Divisions run as magic
    # multiply+shift; the LAST pair needs neither div nor mod (a linear index in
    # range has rem < ext_last; out-of-range lanes decode garbage that the callers'
    # `valid` predication never reads). For a single pair this is linear*stride.
    #
    # INT64: the flat offset can exceed int32 (numel >= 2^31, e.g. a (300000, 8192)
    # reduction). Cast the linear index to Int64 up front so every rem*stride product
    # and accumulation is 64-bit; an int32 product silently wraps negative and reads
    # out of bounds. The returned offset indexes a flat gmem tensor, which expects a
    # 64-bit offset.
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
        assert count < 2**31 and num_o < 2**31  # noqa: S101
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
        # EVERY value baked into the kernel as a const_expr -- now STRUCTURE only.
        # count / num_o / the pair VALUES / in_base / limit / project_n are RUNTIME
        # launch args (grid comes from the output extent), so one compiled kernel
        # serves every geometry sharing this structure: kernel count is
        # O(op x dtype x pair-count), not O(distinct shapes/strides). Callers
        # prepend trait_key and the operand dtypes (not captured here).
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
        # The RUNTIME geometry values, for the per-geometry PLAN cache (which holds
        # the pre-boxed launch args): a repeat geometry skips the Int32/Int64 boxing
        # (~6us/launch) but still shares the structurally-cached compiled kernel.
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
        # Per-block fold bound rb: normally count; with flat_tail (reduce-all
        # stage 1, red stride 1 by construction) clamp to the elements left before
        # `limit` so the overhanging last chunk folds nothing out of range. Runtime
        # value -> the full-wave count n_full is a DYNAMIC loop trip count.
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
            # Stage-2: COMBINE pre-reduced accumulator tuples from the per-field
            # partial buffers. Partials for output o are the contiguous run
            # [obase, obase+count); obase = o*C decoded from kept_pairs (Int64) -- must
            # offset by it, else every row reads row 0's partials (multi-row bug).
            #
            # count is the partial count (for a huge-N reduce-all split, LARGE -- ~1e5).
            # A range_constexpr(ceil(count/block)) unroll scaled the compile with count
            # (count=98125 -> ~384-deep unroll -> ~3s compile; the reduce-all backup's
            # G-chunk was worse). Same fix as the per-axis fold below: a DYNAMIC
            # full-wave loop (all-in-range, so the constant valid=True doesn't trip the
            # IR flattener) + a CONSTEXPR remainder. Compile depth is O(1) in count.
            # Bind trait attributes to locals -- attribute access on `trait` inside a
            # dynamic loop trips the IR flattener (like reduce_fn above); nfields is a
            # small python int so a bare-range comprehension is trace-time unrolled and
            # leaves no trait access in the loop body (range_constexpr can't appear in a
            # cutlass.range loop).
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
            # Per-axis fold (flat_tail included: rb is pre-clamped to the elements
            # before `limit`, so r < rb already implies off < limit -- the old
            # per-element off guard collapsed into the rb clamp above). The trip
            # count n_full is a DYNAMIC value: every full wave is all-in-range, so
            # the loop guard is the python constant True and a cutlass.range loop
            # compiles (a dynamic per-element `valid` would trip the IR flattener).
            # Compile depth is O(1) in count; one predicated remainder pass follows.
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


# ---------------------------------------------------------------------------
# Host plumbing + the geometry chooser. These build ReduceBlock launches; the
# kernel above is the only @cute.kernel in the whole library.
# ---------------------------------------------------------------------------
_cute = _L.cute_tensor
_stream = _L.stream
_compile = (
    _L.compile
)  # _L.compile: cute.compile + options="--enable-tvm-ffi" (fast per-call arg passing)
_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}

_COMPILE_CACHE = {}  # structural key -> compiled kernel (one per cache_sig)
_PLAN = {}  # (structural key, geom_sig) -> (compiled fn, pre-boxed geometry args)


def _cute_list(ts, read_only=False):
    # All K0 operands are 1D flat views (input storage, partials, reshaped outs);
    # the leading (only) dim is marked DYNAMIC so the structural kernel serves any
    # length -- required since the grid reads mOuts[0].shape[0] live. read_only
    # wraps INPUT tensors so a COW input exports without materializing (launch._ro).
    return [_L.cute_tensor_dynM(t, ndim=1, read_only=read_only) for t in ts]


def _quads(pairs):
    # (extent, stride) pairs -> the flat [m, sh, ext, strd, ...] quad list
    # _decode_offset consumes (see there). Runs once per NEW geometry (the boxed
    # result is memoized in _PLAN), so the bit_length/divide cost is off the
    # repeat-launch path.
    out = []
    for ext, strd in pairs:
        m, sh = _magic(ext)
        out += [Int64(m), Int64(sh), Int64(ext), Int64(strd)]
    return out


def _geom_args(op):
    # The RUNTIME geometry of a ReduceBlock launch: magic-division quad lists for
    # the reduced/kept decodes plus the scalar bounds. Everything here was a baked
    # const_expr before; the compiled kernel (keyed on the STRUCTURAL cache_sig
    # alone) now takes these per call. The magic form requires linear indices
    # < 2^31; r and o are Int32 by construction, asserted where count/num_o are set.
    return (
        _quads(op.red_pairs),
        _quads(op.kept_pairs),
        Int32(op.count),
        Int64(op.in_base),
        Int64(op.limit),
        Int64(op.project_n),
    )


def _launch(op, key, ins, outs):
    # Two-level cache: _PLAN memoizes (compiled fn, pre-boxed geometry args) per
    # GEOMETRY (boxing 10 Int32/Int64 costs ~6us -- the dominant repeat-launch
    # overhead); _COMPILE_CACHE dedupes the compile per STRUCTURE (key already ends
    # in cache_sig), so new geometries reuse the kernel and only box once. key[1] is
    # the trait_key (e.g. "sum") -> one tlparse artifact per distinct K0 kernel built.
    plan = _PLAN.get((key, op.geom_sig))
    if plan is None:
        fn = cached_plan(
            _COMPILE_CACHE,
            key,
            lambda: _compile(
                op,
                _cute_list(ins, read_only=True),
                _cute_list(outs),
                *_geom_args(op),
                _stream(),
            ),
            op=f"aten::{key[1]}",
        )
        plan = (fn, _geom_args(op))
        _PLAN[(key, op.geom_sig)] = plan
    fn, geom = plan
    fn(_cute_list(ins, read_only=True), _cute_list(outs), *geom, _stream())


def _ti_pairs(x, out):
    """The kernel's input addressing for ``reduce x into out``, from TensorIterator.
    A dim is REDUCED iff the iterator's output stride along it is 0, else KEPT.
    Returns (red_pairs, kept_pairs) of (extent, input_element_stride). TI handles
    all the broadcast/coalesce/reorder; this is a thin read of its result.

    KEPT dims are ordered by OUTPUT stride ascending (fastest output dim first),
    because block index o is decoded mixed-radix fastest-first (_decode_offset)
    and must land on out.reshape(-1)[o] -- i.e. match the output's flat traversal.
    REDUCED dims need no ordering: the fold visits every reduced element exactly
    once and the combine is commutative (single-dim argmax has one pair anyway)."""
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
    # A dummy output tensor with the reduced dims set to size 1, as reduce_op /
    # _ti_pairs expect (it reads shapes+strides only, no compute). Shared by the
    # fast-path classifier and the K0 fallback so both see the same TI decode.
    return torch.empty(
        [1 if i in red_axes else s for i, s in enumerate(x.shape)],
        device=x.device,
        dtype=x.dtype,
    )


def _flat(x):
    # A 1D stride-1 view over x's ENTIRE underlying storage. TI's element strides
    # are storage-relative, so the kernel indexes THIS (not x.reshape(-1), which
    # for a non-contiguous x would copy + reorder and break the stride math).
    n = max(x.untyped_storage().nbytes() // x.element_size(), 1)
    return torch.as_strided(x, (n,), (1,), storage_offset=0)


# --- Fast-path classification (the ONE source of truth shared by the router below
# and the override cond gate in overrides.py). Operates on the TI-decomposed
# (red_pairs, kept_pairs) so it sees the POST-coalesce geometry, not the raw dim
# arg: e.g. a contiguous 3D last-dim reduction coalesces to a single reduced run +
# a single kept run and is serviceable by the fast ROW kernel after a 2D reshape.
#
# The K0 general kernel is correct for ANY geometry but is ~5-8x slower than aten
# (it does scalar/strided offset-decoded loads). So instead of running K0 for
# non-2D geometries, we (a) RESHAPE the coalescible ones into the fast row/col
# kernels here, and (b) DECLINE the rest to aten via the cond gate. "fast_kind"
# returns which fast path a reduction maps to, or None if only K0 could serve it
# (-> the cond declines, aten takes it). ---


def fast_kind(red_pairs, kept_pairs, nouts, has_index):
    """Which fast kernel serves this TI-decomposed reduction, or None (-> K0/aten).

    "row"     : reduced axis is the single contiguous (stride-1) innermost run;
                kept is a single run. Reshape to (prod(kept), prod(red)), reduce
                last dim -> the row kernels / xcta. Any nouts / index trait.
    "col"     : kept axis is the single contiguous (stride-1) innermost run;
                reduced is a single run. Reshape to (prod(red), prod(kept)),
                reduce dim 0 -> K2. ONLY nouts==1 non-index (K2 is value-only).
    "all"     : no kept dims (full reduction) -> xcta / two-stage. Any trait.
    None      : neither -- only the K0 general kernel could serve it; the cond
                declines to aten instead (K0 is far slower than aten's kernel).

    Both axes must coalesce to a SINGLE run (len==1) so the reduction is a dense
    2D (kept, red) or (red, kept) view. For a contiguous input TI collapses each
    dense axis to one pair, so a genuine row/col/coalescible-n-D case gives exactly
    one red + one kept pair; a transpose / multi-run / gapped layout gives >1 and
    falls to None (aten). The stride-1 pair is the innermost (contiguous) axis and
    decides row vs col.
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


# The one-shot stages a whole row tile, so its tile is ~N*dtype_bytes; it must fit the
# ~228 KB B200 smem budget. Above that, route to the multi-CTA split (which caps each
# chunk's tile). Use a conservative 192 KB so the reduction buffer + slack also fit.
_SMEM_BUDGET = 192 * 1024
# ... and the per-thread LOAD count must stay bounded. The fold walks ceil(N/(tpr*vec)) loads
# per thread; that only gets out of hand when the vector width collapses to 1 (an odd or prime
# N), where it becomes N/tpr. MEASURED, (8, N) sum vs ATen with the bound absent: N=32771 fp32
# 34.1us vs 5.8 (0.17x), N=65537 bf16 76.4 vs 6.0 (0.08x), N=98299 bf16 95.4 vs 7.5 (0.08x).
# Above the bound the cross-CTA split serves those instead, measured 1.93-2.41x of ATen on the
# same shapes. 64 separates every measured good case (N=4099 at tpr 64 -> 64 loads, N=49152 at
# vec 4 -> 48) from every bad one (128, 256). tile.MAX_UNROLL bounds the same quantity inside
# the kernel; this is the gate's copy.
_ONESHOT_MAX_LOADS = 64

# K0 general (correctness-fallback) kernel config, as named DATA. K0 is the any-geometry
# backstop (scalar/strided offset-decoded loads), NOT a perf path, so its knobs are
# occupancy baselines, not a tuned surface. reduce-dim uses _K0_BLOCK threads/block;
# reduce-all uses its own (block, grid_mult) since it routes through the xcta two-stage.
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
    tpr = max(WARP, rt.row_config(N, width, 1).tpr)
    return -(-N // (tpr * vec)) <= _ONESHOT_MAX_LOADS


def _try_fast_row(trait, trait_key, x, out_dtypes, nouts):
    # Fast path for reduction of the CONTIGUOUS last dim of a 2D problem. Sub-paths;
    # returns the result tuple, or None if not handled:
    #   smem-safe, load-bounded N -> one-shot (kernel_rowtile, 1 or 2 outputs)
    #   larger N                  -> fused cross-CTA two-stage (reduce_xcta, 1 or 2)
    # The one-shot needs no index remap (the projected index IS the per-row column); the
    # split rebases each chunk's local column to the global one, so index traits are
    # served at any N. A geometry neither accepts returns None -> K0.
    if x.dim() != 2 or x.stride(-1) != 1:
        return None
    N = x.shape[-1]
    if N < 1:
        return None
    if nouts not in (1, 2):
        return None
    from . import kernel_rowtile as rt

    if _oneshot_ok(x):
        return rt.reduce_row_tile(trait, trait_key, x, out_dtypes, nouts=nouts)
    from . import kernel_xcta as xc

    if nouts == 2:
        # The same fused split as nouts==1, projecting both fields. Without it a
        # few-row/huge-N 2-output reduction lands on K0's one-block-per-row (0.63x of
        # ATen at N=65536, 0.20x at N=131072, for every M).
        return xc.reduce_row_xcta_2out(trait, trait_key, x, out_dtypes)
    res = xc.reduce_row_xcta(trait, trait_key, x, out_dtypes[0])
    if res is None:
        return None  # xcta declined (no divisor split for this N) -> K0 general kernel
    return (res,)


def _as_shape(out, out_shape):
    # Give a reduction's flat output its final n-D shape WITHOUT leaving the result a
    # view. reduce_all allocates its own 1-element buffer, so a plain
    # `.reshape(out_shape)` returns a view whose `_base` is that buffer -- and an aten
    # reduction NEVER aliases, a difference OpInfo's python-ref tests do check.
    # `_as_shape` reshapes in place when it can so the buffer IS the result.
    if tuple(out.shape) == tuple(out_shape):
        return out
    reshaped = out.reshape(out_shape)
    if reshaped._base is None:
        return reshaped
    out.resize_(out_shape)
    return out


def _reduce(trait, trait_key, x, dims, out_dtypes, nouts, block=_K0_BLOCK):
    # General reduction of x over `dims` (int / tuple / None=all), driven by TI.
    # Covers row/column/n-D/transposed/sliced uniformly. Returns nouts tensors.
    # block = K0 threads-per-block (exposed knob); baked into ReduceBlock + cache_sig.
    assert x.is_cuda  # noqa: S101
    red_axes = (
        range(x.dim()) if dims is None else ([dims] if isinstance(dims, int) else dims)
    )
    red_axes = {d % x.dim() for d in red_axes}
    out_shape = [s for i, s in enumerate(x.shape) if i not in red_axes]

    # Single output ELEMENT (every kept extent is 1) -> route to reduce_all (the
    # two-stage split), not the K0 fallback. Reached when a multi-dim `dims` happens to
    # cover all axes, or a 1D reduce (the override's reduce-ALL path calls reduce_all
    # directly, but reduce_dim with an explicit full dim set lands here), and -- the M=1
    # row case -- when the kept axes merely have extent 1: TI collapses those away
    # entirely, so fast_kind reports "all" (see its len(kept_pairs) == 0 arm) and the
    # classify block below cannot serve it. Without this a (1, N) reduce-dim landed on
    # K0's ONE block for the whole row. _as_shape restores out_shape (numel 1, so it is
    # pure metadata). Single output only; a 2-output full reduction is rare and stays on K0.
    if math.prod(out_shape) == 1 and nouts == 1 and x.is_contiguous():
        out = reduce_all(trait, trait_key, x, out_dtypes[0], block=block)
        return (_as_shape(out, out_shape),)

    # Classify the POST-TI-coalesce geometry and route to the fast kernels via a
    # dense 2D reshape. A contiguous n-D reduction whose reduced (or kept) axes are
    # innermost coalesces to a single reduced + single kept run -> serviceable by
    # the row/col kernels after reshaping to (kept, red) / (red, kept). This is why
    # e.g. a contiguous (A,B,C) reduce over dim 2 (or dims (1,2)) goes fast instead
    # of to the ~5-8x-slower K0 general kernel. The override cond declines any
    # geometry that returns None here (only K0 could serve it) so aten takes it;
    # `_reduce` therefore only sees fast_kind in {row, col} on the real (non-
    # declined) call. K0 below stays as the correctness fallback for direct callers
    # and for the case a fast kernel itself declines (e.g. xcta on a prime N).
    if len(out_shape) > 0 and x.is_contiguous():
        red_pairs, kept_pairs = _ti_pairs(x, _probe(x, red_axes))
        has_index = getattr(trait, "has_index", False)
        kind = fast_kind(red_pairs, kept_pairs, nouts, has_index)
        red_n = x.numel() // max(1, math.prod(out_shape))
        if kind == "row":
            x2 = x.reshape(math.prod(out_shape), red_n)
            fast = _try_fast_row(trait, trait_key, x2, out_dtypes, nouts)
            if fast is not None:
                return tuple(o.reshape(out_shape) for o in fast)

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
    # G = number of stage-1 chunks (CTAs). Fill the device to grid_mult waves, capped
    # by the work available. grid_mult is the exposed fan-out knob: more chunks = more
    # parallelism in stage 1 but a larger stage-2 fold. Default 4 (the prior constant).
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(
    trait, trait_key, x, out_dtype, block=_K0_ALL_BLOCK, grid_mult=_K0_ALL_GRID_MULT
):
    return _reduce_all(trait, trait_key, x, [out_dtype], 1, block, grid_mult)[0]


def _reduce_all(trait, trait_key, x, out_dtypes, nouts, block, grid_mult):
    # Full-tensor reduce-all, in preference order: the one-shot row kernel (a single
    # kernel, when the input fits its tile), then the fused cross-CTA two-stage
    # (reduce_xcta, the M=1 case -- it mirrors ATen's ctas_per_output split), then the
    # grid-striding two-stage K0. Index traits (argmax/min) are served throughout: a
    # single row makes each sub-row's global column the flat index, and stage 1
    # accumulates exactly that, so the winner's index needs no remap.
    assert x.is_cuda and x.is_contiguous()  # noqa: S101
    L = x.numel()
    xf = x.reshape(-1)
    # Fits the one-shot tile -> no cross-CTA split is wanted at all. Left to xcta, such
    # an input either lands on C == 1 (stage 2 folds a SINGLE partial in a kernel of its
    # own: ~1.9us of pure launch, 45% of the small-input floor) or, below xcta's
    # 256-element sub-row floor, is declined to the two-stage K0 -- one kernel too many
    # either way. _try_fast_row applies this same gate BEFORE reaching for xcta; the
    # reduce-all path calls xcta directly, so it has to apply it here. Measured 1.3-2.1x
    # over the two-stage across the whole legal band, 1.2-2.1x over ATen.
    x2 = xf.view(1, -1)
    if _oneshot_ok(x2):
        from . import kernel_rowtile as rt

        # The launch is ONE row, so the ladder's row-packing tpr would leave the whole
        # device on a fraction of one CTA -- widen it (see rt.single_row_config, which
        # returns None when the ladder's pick already stands).
        cfg = rt.single_row_config(L, x.element_size() * 8, trait.nfields)
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
    # Too big for the one-shot and xcta declined (prime/poorly-factored L) -> two-stage
    # K0, which grid-strides any L with no reshape (O(1) compile regardless of L) and so
    # still fills the device for a single huge row.
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(1, device=x.device, dtype=d) for d in out_dtypes]

    # Stage 1: 1D input split into G contiguous chunks. Modeled in the general
    # scheme as kept dim (G, chunk) and reduced dim (chunk, 1): obase = o*chunk,
    # off = o*chunk + r, with flat_tail guarding off < L on the last chunk.
    # gidx_from="flat" -> argmax carries the true global flat index.
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

    # Stage 2: fold the G per-field partials in one block, project once. The
    # divisor for mean/etc. is the TRUE element count L, not G. (from_partials
    # ignores red_pairs/kept_pairs; pass trivial ones.) project_n=L is in
    # cache_sig, so distinct L never reuse a stale baked-in divisor.
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
