# Vectorized COLUMN reduction (reduce dim 0 of (M, N) -> (N,)). The "outer" /
# K2 geometry: the KEPT axis (columns, N) is contiguous, so adjacent threads own
# adjacent columns and gmem loads coalesce for free. Each thread carries `vec`
# INDEPENDENT accumulators (one per column in its 128-bit vector) and folds DOWN
# the M rows -- "vectorize along output". No cross-thread reduce along the reduced
# axis when one thread owns a column fully; when M is tiled across thread-rows
# (block.y), the per-column partials combine via smem along y only.
#
# Contrast K1 (inner/row): there a warp COOPERATES on one output via shuffle. Here
# threads are INDEPENDENT across columns -- the dual. This is why K0's scalar-load
# column path was ~0.2x ATen: it never vectorized the contiguous column axis.
#
# M-SPLIT (mirrors ATen setReduceConfig ctas_per_output for OUTER reductions):
# parallelizing only over columns (grid.x) underfills the device when N is small
# (few columns -> few blocks -> e.g. (65536,1024) gave grid.x=4 blocks on 148
# SMs). So when columns alone don't fill the grid we ALSO split the reduced M axis
# across grid.y CTA-rows: block (bx, by) reduces a disjoint row-stripe of its
# columns into a raw partial, then a cheap stage-2 column reduction combines the
# grid_y partials per column and projects once with the TRUE M. grid_y==1 keeps
# the original single-launch path. This is the dual of reduce_xcta's row split.
#
# Trait protocol reused from reduce_traits; the per-column fold is the trait's
# reduce/combine on a tuple accumulator, one tuple PER column-in-vector.

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32

import torch

from .._cutedsl import hw_caps as _hw, launch as _L
from .._cutedsl.plan_cache import cached_plan


_cute = _L.cute_tensor
_compile = _L.compile  # cute.compile + options="--enable-tvm-ffi"
_stream = _L.stream
_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}

# Threads along the column (kept) axis per block. The dispatcher in kernel_general
# uses this to size grid_x when deciding the K2 column path, so it must match the
# block_x the kernel actually launches with -- keep them tied to this constant.
_DEFAULT_BLOCK_X = 64
_DEFAULT_BLOCK_Y = 8


class ColReduce:
    # Grid (ceil(N / (block_x*vec)), grid_y). Block = (block_x, block_y). Thread
    # (tx, ty) in CTA-row `by` owns columns [col0, col0+vec) where
    # col0 = (bx*block_x + tx)*vec, and folds the row-stripe by*block_y+ty,
    # +grid_y*block_y, ... down `rows`. The block_y partials combine via smem; the
    # grid_y CTA-row partials combine in stage 2 (from_partials).
    #
    # final=True  -> project (with true_m) and write one result per column.
    # final=False -> write RAW per-field accumulators to nfields (grid_y, N) gmem
    #                partial buffers (stage 1 of the M-split).
    # from_partials=True -> ingest COMBINES nfields pre-reduced partial buffers
    #                instead of REDUCing raw input (stage 2 of the M-split).
    def __init__(
        self,
        trait,
        rows,
        N,
        vec,
        true_m,
        grid_y=1,
        final=True,
        from_partials=False,
        block_x=_DEFAULT_BLOCK_X,
        block_y=_DEFAULT_BLOCK_Y,
    ):
        self.trait = trait
        self.rows = rows  # rows of THIS stage's input
        self.N = N
        self.vec = vec
        self.true_m = true_m  # original M, the projection divisor
        self.grid_y = grid_y  # CTA-rows splitting the reduced axis
        self.final = final
        self.from_partials = from_partials
        self.block_x = block_x
        self.block_y = block_y
        self.cols_per_block = block_x * vec
        self.grid_x = (N + self.cols_per_block - 1) // self.cols_per_block

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, stream: cuda.CUstream):
        self.kernel(mIns, mOuts).launch(
            grid=[self.grid_x, self.grid_y, 1],
            block=[self.block_x, self.block_y, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list):
        trait = self.trait
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        rows = const_expr(self.rows)
        N = const_expr(self.N)
        vec = const_expr(self.vec)
        by_n = const_expr(self.block_y)
        nf = const_expr(trait.nfields)
        acc_dtype = trait.acc  # accumulator dtype (a compile-time Python class)

        col0 = (bx * const_expr(self.block_x) + tx) * vec
        accs = [trait.init() for _ in range(vec)]
        # Global thread-row partition: CTA-row `by` thread-row `ty` starts at
        # by*block_y+ty and strides by the TOTAL thread-rows grid_y*block_y, so
        # the grid_y CTA-rows tile `rows` disjointly. row_stride collapses to
        # block_y when grid_y==1 (the single-launch path).
        row_stride = const_expr(self.grid_y * self.block_y)
        gty0 = by * by_n + ty
        n_full = const_expr(rows // row_stride)

        if const_expr(not self.from_partials):
            reduce_fn = trait.reduce
            mX = mIns[0]
            frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
            rr = gty0
            for _ in cutlass.range(n_full):
                # Int64 row base: rr*N overflows int32 when rows*N >= 2^31 (large
                # column reductions) -> negative wrap -> illegal access. Cast rr.
                cute.autovec_copy(_row_vec(mX, cutlass.Int64(rr) * N + col0, vec), frag)
                xf = frag.load().to(acc_dtype)
                for v in cutlass.range_constexpr(vec):
                    accs[v] = reduce_fn(accs[v], xf[v], rr, col0 + v < N)
                rr = rr + row_stride
            if const_expr(rows % row_stride != 0):
                ok = rr < rows
                cute.autovec_copy(
                    _row_vec(mX, cutlass.Int64(rr if ok else Int32(0)) * N + col0, vec),
                    frag,
                )
                xf = frag.load().to(acc_dtype)
                for v in cutlass.range_constexpr(vec):
                    accs[v] = reduce_fn(accs[v], xf[v], rr, ok and (col0 + v < N))
        else:
            # Combine pre-reduced partials: nfields buffers, each (rows, N).
            combine_fn = trait.combine
            frags = [
                cute.make_rmem_tensor(cute.make_layout(vec), mIns[f].element_type)
                for f in range(nf)
            ]
            rr = gty0
            for _ in cutlass.range(n_full):
                rb64 = (
                    cutlass.Int64(rr) * N + col0
                )  # Int64 base (rows*N may exceed 2^31)
                for f in cutlass.range_constexpr(nf):
                    cute.autovec_copy(_row_vec(mIns[f], rb64, vec), frags[f])
                for v in cutlass.range_constexpr(vec):
                    part = tuple(frags[f][v] for f in range(nf))
                    merged = combine_fn(accs[v], part)
                    valid = col0 + v < N
                    accs[v] = tuple(
                        (merged[f] if valid else accs[v][f]) for f in range(nf)
                    )
                rr = rr + row_stride
            if const_expr(rows % row_stride != 0):
                ok = rr < rows
                rb64 = cutlass.Int64(rr if ok else Int32(0)) * N + col0
                for f in cutlass.range_constexpr(nf):
                    cute.autovec_copy(_row_vec(mIns[f], rb64, vec), frags[f])
                for v in cutlass.range_constexpr(vec):
                    part = tuple(frags[f][v] for f in range(nf))
                    merged = combine_fn(accs[v], part)
                    valid = ok and (col0 + v < N)
                    accs[v] = tuple(
                        (merged[f] if valid else accs[v][f]) for f in range(nf)
                    )

        # Combine the block_y partials per column via smem (cross thread-row).
        smem = cutlass.utils.SmemAllocator()
        if const_expr(by_n > 1):
            bufs = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_ordered_layout(
                        (by_n, const_expr(self.block_x * vec)), order=(1, 0)
                    ),
                    byte_alignment=8,
                )
                for f in range(nf)
            ]
            for v in cutlass.range_constexpr(vec):
                cidx = tx * vec + v
                for f in cutlass.range_constexpr(nf):
                    bufs[f][(ty, cidx)] = trait.fdtypes[f](accs[v][f])
            cute.arch.barrier()
            for v in cutlass.range_constexpr(vec):
                cidx = tx * vec + v
                merged = trait.init()
                for yy in cutlass.range_constexpr(by_n):
                    part = tuple(bufs[f][(yy, cidx)] for f in range(nf))
                    merged = trait.combine(merged, part)
                accs[v] = merged

        # ty==0 of each (bx, by) block writes its column results. final ->
        # project (true_m divisor) one value per column; not-final -> store RAW
        # per-field accumulators to parts[f][by, col] for the stage-2 combine.
        if const_expr(self.final):
            for v in cutlass.range_constexpr(vec):
                col = col0 + v
                res = mOuts[0].element_type(
                    trait.project(accs[v], acc_dtype(const_expr(self.true_m)))
                )
                if ty == 0 and col < N:
                    mOuts[0][col] = res
        else:
            for v in cutlass.range_constexpr(vec):
                col = col0 + v
                vals = [trait.fdtypes[f](accs[v][f]) for f in range(nf)]
                if ty == 0 and col < N:
                    for f in cutlass.range_constexpr(nf):
                        mOuts[f][by * N + col] = vals[f]


@cute.jit
def _row_vec(mX, base, vec: cutlass.Constexpr):
    # A vec-wide contiguous slice of the flat input starting at `base`, viewed as
    # a (vec,) tensor for autovec_copy.
    return cute.make_tensor(mX.iterator + base, cute.make_layout(vec))


_CACHE = {}


def _aligned(t, align, read_only=False):
    # enable_tvm_ffi: fast torch->tvm-ffi C exchange (~0.8us vs ~3.6us capsule).
    # read_only wraps an INPUT so a COW input exports without materializing (launch._ro).
    w = _L.ReadOnlyTensorWrapper(t) if read_only else t
    ct = cute.runtime.from_dlpack(w, assumed_align=align, enable_tvm_ffi=True)
    ct.element_type = _L.torch2cute[t.dtype]
    return ct


def _choose_config(M, N, vec, hw):
    # Launch config (block_x, block_y, grid_y) for the column reduction.
    #
    # NO-REGRESSION baseline. A dense landscape sweep (characterize_k2.py, f32/f16 x
    # 11 shapes) showed there's real headroom (up to +0.3x on some shapes) BUT no
    # single closed-form rule captures the surface without REGRESSING other shape
    # classes -- every scalar formula trades square/wide against tall-narrow. Since
    # the directive is "don't regress any case," we keep the proven heuristic here
    # and leave the headroom to a future AUTOTUNER (which measures per shape-key and
    # picks the best, so it cannot regress -- the documented follow-up). The sweep
    # data + hw_caps are the inputs that autotuner will use.
    #
    # block_x/block_y fixed at the validated defaults; grid_y from _choose_grid_y:
    # split the M (reduced) axis only when the column-blocks underfill the device,
    # to ~2*sm_count total blocks, capped by available M. sm_count is read from hw
    # so the fill target tracks the GPU (Hopper/Blackwell/Rubin).
    block_x, block_y = _DEFAULT_BLOCK_X, _DEFAULT_BLOCK_Y
    grid_x = -(-N // (block_x * vec))
    gy = _choose_grid_y(M, grid_x, block_y, hw.sm_count)
    return block_x, block_y, gy


def _choose_grid_y(M, grid_x, block_y, sm, min_vpt=64):
    # ctas_per_output for the M (reduced) axis. Split M ONLY when the column-blocks
    # alone leave the device underfilled (splitting adds a gmem partial round-trip,
    # so it only pays when it buys occupancy). Near a full wave (grid_x >= 3/4 sm)
    # don't split. Otherwise target ~2*sm total blocks; cap so each thread-row still
    # folds >= min_vpt rows. sm is hw.sm_count (portable across GPUs).
    if grid_x >= (3 * sm) // 4:
        return 1
    gy = max(1, -(-(2 * sm) // max(grid_x, 1)))
    return max(1, min(gy, max(1, M // (block_y * min_vpt))))


def reduce_col(
    trait,
    trait_key,
    x,
    out_dtype,
    block_x=None,
    block_y=None,
    grid_y=None,
    vec_bits=128,
):
    # Vectorized column reduction (reduce dim 0). x: (M, N) contiguous.
    # block_x/block_y/grid_y default to the HW-parameterized heuristic
    # (_choose_config); pass explicit values to override (used by the autotune
    # landscape sweep characterize_k2.py). vec_bits is the exposed load/store vector
    # width in bits (default 128 = the wide LDG/STG target); 64/256 are the other
    # candidates. gcd with N keeps it a legal divisor for ragged N.
    assert x.dim() == 2 and x.is_cuda and x.stride(-1) == 1  # noqa: S101
    M, N = x.shape
    vec = math.gcd(N, vec_bits // (x.element_size() * 8))
    cbx, cby, cgy = _choose_config(M, N, vec, _hw.caps(x.device))
    block_x = cbx if block_x is None else block_x
    block_y = cby if block_y is None else block_y
    grid_y = cgy if grid_y is None else grid_y
    out = torch.empty(N, device=x.device, dtype=out_dtype)
    xf = x.reshape(-1)
    align = 16 if (N % vec == 0) else x.element_size()

    if grid_y == 1:
        op = ColReduce(
            trait,
            M,
            N,
            vec,
            true_m=M,
            grid_y=1,
            final=True,
            block_x=block_x,
            block_y=block_y,
        )
        xin = _aligned(xf, align, read_only=True)
        key = (
            "col",
            trait_key,
            x.dtype,
            out_dtype,
            M,
            N,
            vec,
            block_x,
            block_y,
            trait.nfields,
        )
        fn = cached_plan(
            _CACHE, key, lambda: _compile(op, [xin], [_cute(out)], _stream())
        )
        fn([xin], [_cute(out)], _stream())
        return out

    # M-split: stage 1 -> grid_y raw partials per column; stage 2 combines them.
    nf = trait.nfields
    parts = [
        torch.empty(grid_y * N, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(nf)
    ]
    op1 = ColReduce(
        trait,
        M,
        N,
        vec,
        true_m=M,
        grid_y=grid_y,
        final=False,
        from_partials=False,
        block_x=block_x,
        block_y=block_y,
    )
    xin = _aligned(xf, align, read_only=True)
    k1 = ("col-s1", trait_key, x.dtype, M, N, vec, grid_y, block_x, block_y, nf)
    f1 = cached_plan(
        _CACHE, k1, lambda: _compile(op1, [xin], [_cute(p) for p in parts], _stream())
    )
    f1([xin], [_cute(p) for p in parts], _stream())

    # Stage 2: column-reduce the (grid_y, N) partials -> (N,), project with M.
    # Partials are Float32/Int32 storage; recompute vec for their element size.
    pvec = math.gcd(N, 128 // (parts[0].element_size() * 8))
    op2 = ColReduce(
        trait,
        grid_y,
        N,
        pvec,
        true_m=M,
        grid_y=1,
        final=True,
        from_partials=True,
        block_x=block_x,
        block_y=block_y,
    )
    palign = 16 if (N % pvec == 0) else parts[0].element_size()
    pin = [_aligned(p, palign, read_only=True) for p in parts]
    k2 = ("col-s2", trait_key, out_dtype, grid_y, N, pvec, block_x, block_y, nf)
    f2 = cached_plan(_CACHE, k2, lambda: _compile(op2, pin, [_cute(out)], _stream()))
    f2(pin, [_cute(out)], _stream())
    return out
