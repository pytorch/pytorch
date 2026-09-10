# General CuteDSL reduction kernel and dispatcher. TensorIterator marks reduced dimensions
# with zero output stride; compile-time (extent, element-stride) lists decode linear indices
# for arbitrary dimensions and layouts. Reduce-all has no kept dimensions. Only pair counts
# are compiled; runtime geometry lets one kernel serve all matching structures.

import math
from collections.abc import Sequence

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch
from torch._tensor_iterator import reduce_op

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, WARP, warp_reduce


# (extent, element-stride) pairs from TensorIterator, fastest dim first.
Pairs = list[tuple[int, int]]


def _decode_offset(linear, divs, strides, npairs):
    # Mixed-radix linear-to-flat decode with only pair count compiled. Div/mod stays Int32
    # because linear < 2**31; only stride products need Int64. Callers omit empty decodes.
    rem = Int32(linear)
    if npairs == 1:
        return cutlass.Int64(rem) * strides[0]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        q, r = divmod(rem, divs[j])
        off = off + cutlass.Int64(r) * strides[j]
        rem = q
    return off + cutlass.Int64(rem) * strides[npairs - 1]


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
        # (extent, input-element-stride) pairs from TensorIterator, fastest first.
        self.red_pairs = tuple(red_pairs)
        self.kept_pairs = tuple(kept_pairs)
        self.npairs_red = len(self.red_pairs)
        self.npairs_kept = len(self.kept_pairs)
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
        self.num_warps = block // WARP

    @property
    def cache_sig(self):
        # Callers prepend trait and dtype; all geometry values remain runtime arguments.
        return (
            self.npairs_red,
            self.npairs_kept,
            self.nouts,
            self.final,
            self.gidx_from,
            self.flat_tail,
            self.ragged_chunk,
            self.from_partials,
            self.block,
            self.trait.nfields,
        )

    @property
    def geom_sig(self):
        # Cache boxed runtime geometry (~6us) separately while sharing the structural kernel.
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
        rexts: list,
        rstrides: list,
        kexts: list,
        kstrides: list,
        count: cutlass.Int32,
        in_base: cutlass.Int64,
        limit: cutlass.Int64,
        project_n: cutlass.Int64,
        stream,
    ):
        # Build V2 divisors inside the MLIR context so .divisor crosses the kernel boundary.
        rdivs = [cute.FastDivmodDivisorV2(e) for e in rexts]
        kdivs = [cute.FastDivmodDivisorV2(e) for e in kexts]
        # Dynamic grid: read the output row count live so one compile serves any M.
        self.kernel(
            mIns,
            mOuts,
            rdivs,
            rstrides,
            kdivs,
            kstrides,
            count,
            in_base,
            limit,
            project_n,
        ).launch(
            grid=[mOuts[0].shape[0], 1, 1], block=[self.block, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mIns: list,
        mOuts: list,
        rdivs: list,
        rstrides: list,
        kdivs: list,
        kstrides: list,
        count: cutlass.Int32,
        in_base: cutlass.Int64,
        limit: cutlass.Int64,
        project_n: cutlass.Int64,
    ):
        """Fold one kept coordinate per block. Decode its base, fold raw values or
        partial tuples, merge threads, then project and store from thread zero.
        """
        trait = self.trait
        tidx, _, _ = cute.arch.thread_idx()
        o, _, _ = cute.arch.block_idx()
        nfields = const_expr(trait.nfields)

        acc = trait.init()
        obase = in_base  # 0 kept pairs (reduce-all) -> in_base alone
        if const_expr(self.npairs_kept > 0):
            obase = in_base + _decode_offset(o, kdivs, kstrides, self.npairs_kept)
        chunk_base = Int32(0)
        rb = count  # flat_tail: clamp so the overhanging last chunk folds nothing out of range
        if const_expr(self.flat_tail):
            left = limit - obase
            c64 = cutlass.Int64(count)
            left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
            zero = cutlass.Int64(0)
            left = left if left > zero else zero  # noqa: FURB136 -- no DSL builtin max
            rb = cutlass.Int32(left)
        elif const_expr(self.ragged_chunk):
            # Clamp a short final chunk to its reduced run. The fastest-varying kept
            # pair identifies the chunk in steps, for either row or column splits.
            _, cc = divmod(Int32(o), kdivs[0])
            c = cutlass.Int64(cc)
            cnt = cutlass.Int64(count)
            chunk_base = Int32(c * cnt)  # this chunk's first step, for gidx
            left = limit - c * cnt
            c64 = cutlass.Int64(count)
            left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
            zero = cutlass.Int64(0)
            left = left if left > zero else zero  # noqa: FURB136 -- no DSL builtin max
            rb = cutlass.Int32(left)
        n_full = rb // const_expr(self.block)
        reduce_fn = trait.reduce  # local bind: attribute access trips a dyn loop
        acc_dtype = trait.acc  # accumulator dtype (a compile-time Python class)
        if const_expr(self.from_partials):
            # Stage 2 uses a dynamic loop; static unrolling took about 3s at 1e5 partials.
            combine_fn = trait.combine
            fdtypes = trait.fdtypes
            nf = const_expr(nfields)
            r = tidx
            for _ in cutlass.range(n_full):
                rr = obase + cutlass.Int64(r)
                part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
                acc = combine_fn(acc, part)
                r = r + const_expr(self.block)
            # Runtime count always emits a predicated remainder; full waves disable every lane.
            valid = r < rb
            rr = (obase + cutlass.Int64(r)) if valid else in_base
            part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
            merged = combine_fn(acc, part)
            acc = tuple((merged[f] if valid else acc[f]) for f in range(nf))
        else:
            # rb is pre-clamped, so r < rb is the only bounds check. A dynamic trip count
            # keeps compile depth constant in the extent.
            base_r = tidx
            for _ in cutlass.range(n_full):
                # Inline the offset to avoid a spurious loop-carried value. "flat" gidx
                # repeats its single-pair decode, which is one multiply.
                if const_expr(self.gidx_from == "flat"):
                    acc = reduce_fn(
                        acc,
                        acc_dtype(
                            mIns[0][
                                obase
                                + _decode_offset(
                                    base_r, rdivs, rstrides, self.npairs_red
                                )
                            ]
                        ),
                        Int32(
                            obase
                            + _decode_offset(base_r, rdivs, rstrides, self.npairs_red)
                        ),
                        True,
                    )
                elif const_expr(self.gidx_from == "chunk"):
                    # Rebase the chunk-local index; inline it to avoid a loop-carried value.
                    acc = reduce_fn(
                        acc,
                        acc_dtype(
                            mIns[0][
                                obase
                                + _decode_offset(
                                    base_r, rdivs, rstrides, self.npairs_red
                                )
                            ]
                        ),
                        chunk_base + base_r,
                        True,
                    )
                else:
                    acc = reduce_fn(
                        acc,
                        acc_dtype(
                            mIns[0][
                                obase
                                + _decode_offset(
                                    base_r, rdivs, rstrides, self.npairs_red
                                )
                            ]
                        ),
                        base_r,
                        True,
                    )
                base_r = base_r + const_expr(self.block)
            # Invalid lanes read in_base; obase may exceed the input for an empty tail chunk.
            valid = base_r < rb
            off = obase + _decode_offset(base_r, rdivs, rstrides, self.npairs_red)
            off_s = off if valid else in_base
            val = acc_dtype(mIns[0][off_s])
            # gidx is the Int32 arg-reduction position; "flat" is the reduce-all offset.
            if const_expr(self.gidx_from == "flat"):
                acc = reduce_fn(acc, val, Int32(off_s), valid)
            elif const_expr(self.gidx_from == "chunk"):
                acc = reduce_fn(acc, val, chunk_base + base_r, valid)
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
            # Project once using the true reduction size, not stage 2's partial count.
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


# Host-side geometry and plans; no kernels below.
_stream = _L.stream
# _L.compile_kernel uses fake operands and tvm-ffi, avoiding per-call tensor wrapping.
_compile = _L.compile_kernel
_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}

_COMPILE_CACHE = {}  # structural key -> compiled kernel (one per cache_sig)
_PLAN = {}  # (structural key, geom_sig) -> (compiled fn, pre-boxed geometry args)


def _fakes(ts: list[torch.Tensor]) -> list:
    # Dynamic flat descriptors let one structural kernel serve every length.
    return [_L.fake_compact(torch2cute[t.dtype], (_L.sym(),)) for t in ts]


def _operands(ts: list[torch.Tensor], read_only: bool = False) -> list:
    # Pass real tensors; read_only() prevents COW input materialization during export.
    return [_L.read_only(t) for t in ts] if read_only else list(ts)


def _exts(pairs: Pairs) -> list:
    # The launch builds FastDivmod objects for these extents inside its MLIR context.
    return [Int32(ext) for ext, _ in pairs]


def _strides(pairs: Pairs) -> list:
    return [Int64(strd) for _, strd in pairs]


def _geom_args(op):
    # Runtime geometry: both decodes' extents/strides and scalar bounds.
    return (
        _exts(op.red_pairs),
        _strides(op.red_pairs),
        _exts(op.kept_pairs),
        _strides(op.kept_pairs),
        Int32(op.count),
        Int64(op.in_base),
        Int64(op.limit),
        Int64(op.project_n),
    )


def _launch(op, key, ins, outs):
    # _PLAN caches boxed geometry (~6us); _COMPILE_CACHE deduplicates structural kernels.
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


def _ti_pairs(x: torch.Tensor, out: torch.Tensor) -> tuple[Pairs, Pairs]:
    """Return reduced and kept (extent, input-stride) pairs from TensorIterator.
    Zero output stride marks reductions; kept dimensions sort by output stride for
    fastest-first block decoding. Reduced-dimension order is irrelevant.
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
    n = max(x.untyped_storage().nbytes() // x.element_size(), 1)
    return torch.as_strided(x, (n,), (1,), storage_offset=0)


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
    tpr = max(WARP, rt.row_config(N, width).tpr)
    return -(-N // (tpr * vec)) <= _ONESHOT_MAX_LOADS


def _try_fast_row(
    trait, trait_key: str, x: torch.Tensor, out_dtypes: list, nouts: int
) -> tuple | None:
    # Fast contiguous 2D last-dimension path. It needs no index remap, so index traits work.
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
    if reshaped._base is None:
        return reshaped
    out.resize_(out_shape)
    return out


def _two_stage_row(trait, trait_key, x, out_dtypes, nouts, block=_K0_ALL_BLOCK):
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
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
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


def _reduce(trait, trait_key, x, dims, out_dtypes, nouts, block=_K0_BLOCK):
    # TI drives all dimensions and layouts through this path; return nouts tensors.
    if not x.is_cuda:
        raise AssertionError(f"need a CUDA input, got {x.device}")
    red_axes = (
        range(x.dim()) if dims is None else ([dims] if isinstance(dims, int) else dims)
    )
    red_axes = {d % x.dim() for d in red_axes}
    out_shape = [s for i, s in enumerate(x.shape) if i not in red_axes]

    # One-output reductions use reduce_all's split instead of one block for the entire row.
    # This includes full dims and M=1 rows whose size-one kept axes TI removes.
    if math.prod(out_shape) == 1 and nouts == 1 and x.is_contiguous():
        out = reduce_all(trait, trait_key, x, out_dtypes[0], block=block)
        return (_as_shape(out, out_shape),)

    # Reshape post-TI contiguous innermost reductions onto a fast kernel; general remains
    # the fallback for direct calls and declines.
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
            # Splitting the reduced axis supplies parallelism for tall-narrow inputs:
            # 7.24x, 2.53x, and 1.49x of ATen at (65536, 256), (16384, 1024), and (4096, 4096).
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
    # Choose G chunks to fill grid_mult waves, trading stage-1 parallelism for stage-2 work.
    by_work = (L + block - 1) // block
    return max(1, min(by_work, sm_count * grid_mult))


def reduce_all(
    trait, trait_key, x, out_dtype, block=_K0_ALL_BLOCK, grid_mult=_K0_ALL_GRID_MULT
):
    return _reduce_all(trait, trait_key, x, [out_dtype], 1, block, grid_mult)[0]


def _reduce_all(trait, trait_key, x, out_dtypes, nouts, block, grid_mult):
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
    # If xcta declines, grid-stride any L without reshaping or extent-dependent compile cost.
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    G = _grid_size(L, block, sm, grid_mult)
    chunk = (L + G - 1) // G

    parts = [
        torch.empty(G, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
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
