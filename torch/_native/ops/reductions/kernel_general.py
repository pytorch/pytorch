# General CuTeDSL reduction kernel (K0) + the dispatcher for the whole fold-
# reduction taxonomy. K0 is one @cute.kernel that handles ANY geometry (row,
# column, n-D, transposed, sliced, reduce-all) via a TensorIterator-derived
# offset decode -- the correctness floor and fallback. _reduce() dispatches the
# perf-critical cases to specialized kernels and falls back to K0 otherwise:
#   contiguous last-dim, smem-fits   -> kernel_row.reduce_row   (K1, vectorized)
#   contiguous last-dim, larger N    -> reduce_xcta.reduce_row_xcta  (cross-CTA)
#   dim-0 (columns), wide N          -> kernel_col.reduce_col     (K2, vectorized)
#   everything else                  -> K0 here
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
# const_expr policy flags (all compiled away -> specialized SASS, no runtime br):
#   red_pairs    : tuple of (extent, stride) for reduced dims (input elements).
#   kept_pairs   : tuple of (extent, stride) for kept dims (input elements).
#   in_base      : flat input element offset of output coordinate 0.
#   nouts        : 1 or 2 result tensors (var_mean / max.dim are 2).
#   gidx_from    : "r" -> argmax index is the linear reduction index r (per-axis
#                  reductions); "flat" -> the global flat input offset (reduce-all).
#   final        : True  -> project the accumulator, store nouts result(s);
#                  False -> store the RAW accumulator fields to per-field gmem
#                           partial buffers (cross-CTA stage 1).
#   from_partials: True  -> per-thread step COMBINES pre-reduced accumulator
#                  tuples read from nfields partial buffers (stage 2);
#                  False -> it REDUCEs raw scalar inputs.
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


def _decode_offset(linear, pairs):
    # Mixed-radix decode of a linear index into a flat element offset over the
    # given (extent, stride) pairs. pairs are ordered fastest-varying first, so
    # successive divisions peel off each dim. Pure const_expr structure (the loop
    # and the pairs are compile-time); for a single pair this is linear*stride.
    #
    # INT64: the flat offset can exceed int32 (numel >= 2^31, e.g. a (300000, 8192)
    # reduction). Cast the linear index to Int64 up front so every rem*stride product
    # and accumulation is 64-bit; an int32 product silently wraps negative and reads
    # out of bounds. The returned offset indexes a flat gmem tensor, which expects a
    # 64-bit offset.
    off = cutlass.Int64(0)
    rem = cutlass.Int64(linear)
    for ext, strd in pairs:
        if const_expr(len(pairs) == 1):
            off = rem * const_expr(strd)
        else:
            off = off + (rem % const_expr(ext)) * const_expr(strd)
            rem = rem // const_expr(ext)
    return off


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
        dyn_num_o=False,
    ):
        self.trait = trait
        self.count = count  # elements reduced per output (= prod red exts)
        self.num_o = num_o  # number of outputs / blocks (= prod kept exts)
        # dyn_num_o: drive the grid from mOuts[0].shape[0] at launch instead of the
        # baked num_o, and exclude num_o from cache_sig. Valid ONLY for a SINGLE kept
        # dim (the output-row axis): _decode_offset then ignores the extent (single
        # pair -> rem*stride), so the kernel is correct for any num_o. Lets one
        # compiled kernel serve any M (e.g. varying batch size) with no recompile.
        self.dyn_num_o = dyn_num_o
        # (extent, input-element-stride) pairs from TensorIterator, fastest first.
        self.red_pairs = tuple(red_pairs)
        self.kept_pairs = tuple(kept_pairs)
        self.in_base = in_base  # flat input offset of output coordinate 0
        self.limit = limit if limit is not None else count  # ragged tail bound
        # The N passed to project (mean's divisor). = count single-stage; = L for
        # reduce-all stage 2 (where count is the partial count G, not L).
        self.project_n = project_n if project_n is not None else count
        self.nouts = nouts
        self.final = final
        self.gidx_from = gidx_from  # "r" (per-axis index) or "flat" (reduce-all)
        self.flat_tail = flat_tail  # extra off<limit guard for reduce-all stage 1
        self.from_partials = from_partials
        self.block = block
        self.num_warps = block // WARP
        self.iters = (count + block - 1) // block

    @property
    def cache_sig(self):
        # EVERY value baked into the kernel as a const_expr. The compile cache key
        # must include all of these: each distinct combination compiles a distinct
        # kernel. (Missing one -- e.g. project_n -- silently reuses a kernel with a
        # stale baked-in constant; that was a real bug.) Callers prepend trait_key
        # and the operand dtypes (not captured here).
        # num_o is excluded when dynamic (grid comes from the output extent at launch,
        # not baked) so one kernel serves any M; dyn_num_o is in the sig instead.
        num_o_sig = "dynM" if self.dyn_num_o else self.num_o
        return (
            self.count,
            num_o_sig,
            self.red_pairs,
            self.kept_pairs,
            self.in_base,
            self.limit,
            self.project_n,
            self.nouts,
            self.final,
            self.gidx_from,
            self.flat_tail,
            self.from_partials,
            self.block,
            self.trait.nfields,
        )

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, stream):
        # Dynamic grid: read the output row count live so one compile serves any M.
        grid_o = mOuts[0].shape[0] if const_expr(self.dyn_num_o) else self.num_o
        self.kernel(mIns, mOuts).launch(
            grid=[grid_o, 1, 1], block=[self.block, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list):
        trait = self.trait
        tidx, _, _ = cute.arch.thread_idx()
        o, _, _ = cute.arch.block_idx()
        count = const_expr(self.count)
        nfields = const_expr(trait.nfields)

        acc = trait.init()
        # Base flat input offset for this block's KEPT coordinate (decode o
        # against the kept dims). const_expr in/0-kept (reduce-all) -> just in_base.
        obase = const_expr(self.in_base) + _decode_offset(o, self.kept_pairs)
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
            n_full = const_expr(count // self.block)
            r = tidx
            for _ in cutlass.range(n_full):
                rr = obase + cutlass.Int64(r)
                part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
                acc = combine_fn(acc, part)
                r = r + const_expr(self.block)
            if const_expr(count % self.block != 0):
                valid = r < count
                rr = (
                    (obase + cutlass.Int64(r))
                    if valid
                    else cutlass.Int64(const_expr(self.in_base))
                )
                part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
                merged = combine_fn(acc, part)
                acc = tuple((merged[f] if valid else acc[f]) for f in range(nf))
        elif const_expr(self.flat_tail):
            # Reduce-all stage 1: a full wave can still have off >= L on the last
            # chunk (G*chunk > L), so its guard is genuinely dynamic per element ->
            # keep the constexpr loop. count here is a bounded chunk size, not the
            # whole reduction, so the unroll stays small.
            for i in cutlass.range_constexpr(self.iters):
                r = tidx + i * self.block
                off = obase + _decode_offset(r, self.red_pairs)  # Int64 (obase Int64)
                valid = (r < count) and (off < const_expr(self.limit))
                # both ifexp branches Int64 (off is Int64); in_base fits in int32 but
                # the type must match.
                off_s = off if valid else cutlass.Int64(const_expr(self.in_base))
                val = acc_dtype(mIns[0][off_s])
                # gidx is the argmax/flat index fed to the trait (Int32 domain): for
                # reduce-all flat indexing the global offset fits int32 here because
                # flat_tail is only used per-chunk; cast back to Int32.
                gidx = Int32(off_s) if const_expr(self.gidx_from == "flat") else r
                acc = reduce_fn(acc, val, gidx, valid)
        else:
            # Per-axis fold. ceil(count/block) used to be a CONSTEXPR loop -> unroll
            # depth scaled with count (count=1M gave an 8192-deep unroll that barely
            # compiled). Split into:
            #   (a) FULL waves [0, count//block): every thread is in range, so valid
            #       is the python constant True -> a DYNAMIC loop compiles (a dynamic
            #       `valid` would trip the IR flattener; a constant one does not).
            #   (b) a CONSTEXPR remainder pass for the < block leftover, predicated.
            # Compile depth is now O(1) regardless of count. (Only valid when the
            # full-wave guard really is all-true, i.e. NOT the flat_tail case above.)
            n_full = const_expr(count // self.block)
            base_r = tidx
            for _ in cutlass.range(n_full):
                # Inline the offset (no intermediate name that the DSL would treat
                # as loop-carried across iterations). acc and base_r are the only
                # carried values; both are initialized before the loop.
                acc = reduce_fn(
                    acc,
                    acc_dtype(mIns[0][obase + _decode_offset(base_r, self.red_pairs)]),
                    base_r,
                    True,
                )
                base_r = base_r + const_expr(self.block)
            if const_expr(count % self.block != 0):
                r = const_expr(n_full * self.block) + tidx
                valid = r < count
                rs = r if valid else Int32(0)
                acc = reduce_fn(
                    acc,
                    acc_dtype(mIns[0][obase + _decode_offset(rs, self.red_pairs)]),
                    r,
                    valid,
                )

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
            # reduce-all stage 2, where count is just the partial count G).
            result = trait.project(acc, acc_dtype(const_expr(self.project_n)))
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
_PART_TORCH = {
    Float32: torch.float32,
    Float64: torch.float64,
    Int32: torch.int32,
    Int64: torch.int64,
}

_COMPILE_CACHE = {}


def _cute_list(ts, read_only=False):
    # read_only wraps INPUT tensors so a COW input exports without materializing
    # (see launch._ro); outputs are written by the kernel and stay writable.
    return [_cute(t, read_only=read_only) for t in ts]


def _compiled(op, key, ins, outs):
    return cached_plan(
        _COMPILE_CACHE,
        key,
        lambda: _compile(
            op, _cute_list(ins, read_only=True), _cute_list(outs), _stream()
        ),
    )


def _launch(op, key, ins, outs):
    fn = _compiled(op, key, ins, outs)
    fn(_cute_list(ins, read_only=True), _cute_list(outs), _stream())


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
                last dim -> K1 / xcta. Serves any nouts / index trait.
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


# The one-shot K1 stages a whole row tile in smem, so its tile is ~N*dtype_bytes;
# it must fit the ~228 KB B200 smem budget. Above that, route to the multi-CTA
# split (which caps each chunk's smem tile). Use a conservative 192 KB so the
# reduction-buffer + alignment slack also fit.
_SMEM_BUDGET = 192 * 1024


def _oneshot_ok(x):
    # One-shot K1 smem tile fits? (tile ~ N elements of the input dtype).
    return x.shape[-1] * x.element_size() <= _SMEM_BUDGET


def _try_fast_row(trait, trait_key, x, out_dtypes, nouts):
    # Fast path for reduction of the CONTIGUOUS last dim of a 2D problem. Sub-paths;
    # returns the result tuple, or None if not handled:
    #   nouts==2, smem-safe N            -> one-shot K1 two-output (reduce_row_2out)
    #   nouts==1, smem-safe N            -> one-shot K1 (reduce_row)
    #   nouts==1, larger N, non-index op -> single-kernel cross-CTA (reduce_xcta)
    # The two-output one-shot path serves max.dim/min.dim (value + index): no split,
    # so the projected index is the true per-row column. nouts==2 larger-N and
    # index-trait larger-N still fall to the general kernel (no index-aware split).
    if x.dim() != 2 or x.stride(-1) != 1:
        return None
    N = x.shape[-1]
    if N < 1:
        return None
    from . import kernel_row as row

    if nouts == 2:
        if _oneshot_ok(x):
            return row.reduce_row_2out(trait, trait_key, x, out_dtypes)
        return None
    if nouts != 1:
        return None
    if _oneshot_ok(x):
        return (row.reduce_row(trait, trait_key, x, out_dtypes[0]),)
    # Larger N: two-stage cross-CTA reduction (one fused launch). Index traits
    # (argmax/argmin) are now served too: stage-1 accumulates the GLOBAL column
    # index (index_chunks=C), so stage-2 combine over the C partials picks the true
    # global winner with no remap. (max.dim/min.dim are nouts==2 and handled by the
    # one-shot branch above; they do not reach here.)
    from . import kernel_xcta as xc

    res = xc.reduce_row_xcta(trait, trait_key, x, out_dtypes[0])
    if res is None:
        return None  # xcta declined (prime/poorly-factored N) -> use K0 general kernel
    return (res,)


def _reduce(trait, trait_key, x, dims, out_dtypes, nouts, block=128):
    # General reduction of x over `dims` (int / tuple / None=all), driven by TI.
    # Covers row/column/n-D/transposed/sliced uniformly. Returns nouts tensors.
    # block = K0 threads-per-block (exposed knob); baked into ReduceBlock + cache_sig.
    assert x.is_cuda  # noqa: S101
    red_axes = (
        range(x.dim()) if dims is None else ([dims] if isinstance(dims, int) else dims)
    )
    red_axes = {d % x.dim() for d in red_axes}
    has_index = getattr(trait, "has_index", False)
    out_shape = [s for i, s in enumerate(x.shape) if i not in red_axes]

    # Full reduction (every axis reduced -> scalar): route to reduce_all (xcta /
    # two-stage), not the K0 fallback. Reached when a multi-dim `dims` happens to
    # cover all axes, or a 1D reduce (the override's reduce-ALL path calls reduce_all
    # directly, but reduce_dim with an explicit full dim set lands here). Single
    # output only; a 2-output full reduction is rare and stays on K0.
    if len(out_shape) == 0 and nouts == 1:
        return (reduce_all(trait, trait_key, x, out_dtypes[0], block=block),)

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
        kind = fast_kind(red_pairs, kept_pairs, nouts, has_index)
        red_n = x.numel() // max(1, math.prod(out_shape))
        if kind == "row":
            x2 = x.reshape(math.prod(out_shape), red_n)
            fast = _try_fast_row(trait, trait_key, x2, out_dtypes, nouts)
            if fast is not None:
                return tuple(o.reshape(out_shape) for o in fast)
        elif kind == "col":
            from . import kernel_col as cv

            x2 = x.reshape(red_n, math.prod(out_shape))
            out = cv.reduce_col(trait, trait_key, x2, out_dtypes[0])
            return (out.reshape(out_shape),)

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
    # The kernel's input operand is _flat(x) -- a view over x's ENTIRE
    # storage -- and the cute wrap bakes its length statically. cache_sig
    # covers the TI geometry but not the storage size, so two calls with
    # identical geometry over different-sized buffers would collide and
    # feed the second a kernel compiled for the first's flat length
    # (tvm-ffi shape mismatch; hit by conv backward's bias-grad sums in
    # the OpInfo noncontiguous sweep). Key the flat length explicitly.
    flat = _flat(x)
    key = ("reduce", trait_key, x.dtype, tuple(out_dtypes), flat.numel()) + op.cache_sig
    _launch(op, key, [flat], [o.reshape(-1) for o in outs])
    return tuple(outs)


def reduce_dim(trait, trait_key, x, dims, out_dtype, block=128):
    return _reduce(trait, trait_key, x, dims, [out_dtype], 1, block=block)[0]


def reduce_dim2(trait, trait_key, x, dims, out_dtypes, block=128):
    return _reduce(trait, trait_key, x, dims, list(out_dtypes), 2, block=block)


# --- Back-compat shims: the existing harnesses call reduce_row/col/row2 on 2D
# inputs. These now route through the TI-driven general path (dim=-1 for row,
# dim=0 for column). The old x.t() geometry hack is gone -- TI handles strides.


def reduce_row(trait, trait_key, x, out_dtype, block=128):
    assert x.dim() == 2  # noqa: S101
    return reduce_dim(trait, trait_key, x, -1, out_dtype, block=block)


def reduce_row_2out(trait, trait_key, x, out_dtypes, block=128):
    assert x.dim() == 2  # noqa: S101
    return reduce_dim2(trait, trait_key, x, -1, out_dtypes, block=block)


def reduce_col(trait, trait_key, x, out_dtype, block=128, **_legacy):
    # _legacy absorbs the old ColReduction block_x/block_y knobs (now irrelevant:
    # the unified kernel parallelizes one block per kept coordinate).
    assert x.dim() == 2  # noqa: S101
    return reduce_dim(trait, trait_key, x, 0, out_dtype, block=block)


def _grid_size(L, block, sm_count, grid_mult=4):
    # G = number of stage-1 chunks (CTAs). Fill the device to grid_mult waves, capped
    # by the work available. grid_mult is the exposed fan-out knob: more chunks = more
    # parallelism in stage 1 but a larger stage-2 fold. Default 4 (the prior constant).
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(trait, trait_key, x, out_dtype, block=256, grid_mult=4):
    # Full-tensor reduce-all. Routes through the two-stage cross-CTA path (reduce_xcta,
    # the M=1 case) -- it mirrors ATen's ctas_per_output split. Index traits
    # (argmax/min) are served too: the (1, L) -> (C, L/C) reshape makes each sub-row's
    # global column the flat index, and stage-1's index_chunks=C accumulates exactly
    # that, so the winner's index is the true global flat index with no remap.
    assert x.is_cuda and x.is_contiguous()  # noqa: S101
    L = x.numel()
    xf = x.reshape(-1)
    from . import kernel_xcta as xc

    res = xc.reduce_row_xcta(trait, trait_key, xf, out_dtype, flatten=True)
    if res is not None:
        return res
    # xcta declined (prime/poorly-factored L) -> fall through to two-stage K0,
    # which grid-strides any L with no reshape (O(1) compile regardless of L).
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    out = torch.empty(1, device=x.device, dtype=out_dtype)

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
        nouts=1,
        final=True,
        block=block,
    )
    # Key on the partial-buffer dtypes the stage-2 kernel binds to, not just
    # out_dtype: for index traits out_dtype is the int32 INDEX and does not track
    # the value accumulator, so fp32 and fp64 argmax would otherwise collide on one
    # cached kernel (the fp32 one then rejects fp64 partials).
    part_dtypes = tuple(p.dtype for p in parts)
    s2_key = ("all2", trait_key, out_dtype, part_dtypes) + s2.cache_sig
    _launch(s2, s2_key, parts, [out])
    return out.reshape(())
