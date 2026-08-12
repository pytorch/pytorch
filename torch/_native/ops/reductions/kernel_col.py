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
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch

from .._cutedsl import hw_caps as _hw, launch as _L
from .._cutedsl.plan_cache import cached_plan


_cute = _L.cute_tensor
_compile = _L.compile  # cute.compile + options="--enable-tvm-ffi"
_stream = _L.stream
# Accumulator cute dtype -> the torch dtype of K2's stage-1 partial buffers. Int64 is here
# because aten accumulates INTEGER reductions in int64 (gpu_reduce_kernel<scalar_t,
# int64_t>), so an int trait's partials need an int64 buffer; kernel_general and
# kernel_xcta already listed it, and its absence here made K2 the one path that KeyError'd
# on an integer reduction.
_PART_TORCH = {
    Float32: torch.float32,
    Float64: torch.float64,
    Int32: torch.int32,
    Int64: torch.int64,
}

# Threads along the column (kept) axis per block. The dispatcher in kernel_general
# uses this to size grid_x when deciding the K2 column path, so it must match the
# block_x the kernel actually launches with -- keep them tied to this constant.
_DEFAULT_BLOCK_X = 64
_DEFAULT_BLOCK_Y = 8

# grid_y (M-split) device-fill heuristic parameters, as named DATA. The M axis is split
# ONLY when the column-blocks alone underfill the device -- splitting adds a gmem partial
# round-trip, so it only pays when it buys occupancy.
_NEAR_WAVE_FRAC = (3, 4)  # grid_x >= 3/4 * sm -> already near a full wave, don't split
_FILL_TARGET_MULT = 2  # otherwise target ~2 * sm total blocks
_MIN_VPT = 64  # cap the split so each thread-row still folds >= this many rows
# load/store vector width (bits) by dtype element width. A dense graph-mode landscape
# (6 shapes x fp32/bf16) showed NARROW dtypes (bf16/fp16, <=16b) want 64-bit vectors on
# EVERY shape (up to ~2x vs the old 128 default), while fp32 keeps 128 (64 wins on most
# fp32 shapes but REGRESSES the wide-N ones, so 128 stays the no-regression fp32 anchor).
_VEC_BITS_NARROW, _VEC_BITS_WIDE, _NARROW_MAX_BITS = 64, 128, 16


class _ColConfig(NamedTuple):
    # K2 column-reduce knob set. reduce_col() knobs left None are filled from
    # _choose_config; explicit values (autotuner / per-machine retune) override per
    # field. This is K2's OWN config (row/xcta/K0 have differently-shaped knobs).
    block_x: int = _DEFAULT_BLOCK_X
    block_y: int = _DEFAULT_BLOCK_Y
    grid_y: int = 1
    vec_bits: int = _VEC_BITS_WIDE


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
    # Only STRUCTURE is compiled in (see cache_sig); the geometry VALUES
    # (rows / N / true_m / grid_y) are RUNTIME launch args, so one compiled kernel
    # serves every (M, N) sharing a (vec-class, block-shape, path) structure.
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
        m_floor=0,
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
        # M-bucket floor: the host guarantees rows >= m_floor for every launch, so
        # the first m_floor // row_stride fold waves run UNGUARDED with a compile-
        # time trip count -- that is where the strength reduction / load pipelining
        # live (a fully-guarded dynamic loop cost the M-split shapes ~20%: B200
        # 65536x1024, 56 vs 46us). The remainder (< half the fold for half-octave
        # buckets) is a dynamic loop + one predicated tail. m_floor=0 degenerates
        # to the fully-dynamic loop.
        self.m_floor = m_floor

    @property
    def cache_sig(self):
        # Everything baked into the kernel as const_expr. N / grid_y / m_floor are
        # baked for the fold-loop strength reduction and unguarded-wave count (see
        # the kernel notes); the host key adds N and the M-bucket explicitly, so
        # this stays the block/path structure.
        return (
            self.vec,
            self.block_x,
            self.block_y,
            self.grid_y,
            self.m_floor,
            self.final,
            self.from_partials,
            self.trait.nfields,
        )

    def geom_args(self):
        # The runtime geometry, pre-boxable (memoized by the host next to the fn).
        # Only the M-derived values are runtime (M is the high-churn batch axis);
        # N and grid_y are compile-time (both key the plan).
        return (Int32(self.rows), Int64(self.true_m))

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOuts: list,
        rows: cutlass.Int32,
        true_m: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(mIns, mOuts, rows, true_m).launch(
            grid=[self.grid_x, self.grid_y, 1],
            block=[self.block_x, self.block_y, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIns: list,
        mOuts: list,
        rows: cutlass.Int32,
        true_m: cutlass.Int64,
    ):
        trait = self.trait
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        vec = const_expr(self.vec)
        by_n = const_expr(self.block_y)
        nf = const_expr(trait.nfields)
        acc_dtype = trait.acc  # accumulator dtype (a compile-time Python class)
        # N and row_stride are STATIC (both key the plan; see reduce_col) -- a
        # runtime N leaves a 64-bit multiply the compiler can't strength-reduce in
        # the hot fold and a runtime row_stride costs the M-split shapes ~40%
        # (measured on B200). Only rows/true_m (the M-derived values) are RUNTIME:
        # M is the high-churn axis (batch size) and must never recompile.
        N = const_expr(self.N)
        N64 = cutlass.Int64(N)
        row_stride = const_expr(self.grid_y * self.block_y)

        col0 = (bx * const_expr(self.block_x) + tx) * vec
        accs = [trait.init() for _ in range(vec)]
        # Global thread-row partition: CTA-row `by` thread-row `ty` starts at
        # by*block_y+ty and strides by the TOTAL thread-rows grid_y*block_y, so the
        # grid_y CTA-rows tile `rows` disjointly. TWO-PHASE fold (the M-bucket):
        #   A: n_lo = m_floor // row_stride full waves, UNGUARDED, compile-time
        #      trip count (rows >= m_floor is the host's bucket guarantee) -- this
        #      is where the strength reduction and load pipelining live;
        #   B: a guarded DYNAMIC loop from there up to the runtime rows.
        # For a bucketed launch B is <= 25% of the fold (half-octave bucket); for
        # m_floor=0 (unbucketed callers) A is empty and B is the whole fold.
        gty0 = by * by_n + ty
        n_lo = const_expr(self.m_floor // (self.grid_y * self.block_y))

        if const_expr(not self.from_partials):
            reduce_fn = trait.reduce
            mX = mIns[0]
            frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
            rr = gty0
            # xf must EXIST before the dynamic phase-B loop (a name first assigned
            # inside a dynamic `for` trips the IR flattener when n_lo == 0 and
            # phase A never traces). Dead value; frag is written before every use.
            xf = frag.load().to(acc_dtype)
            # Phase A: n_lo unguarded waves at a COMPILE-TIME trip count (the
            # M-bucket floor guarantee) -- unrolled and pipelined. Phase B:
            # dynamic full waves from the floor to the runtime rows, then one
            # predicated remainder. (A constexpr guarded tail to the bucket
            # ceiling was measured SLOWER than the dynamic phase B.)
            for _ in cutlass.range_constexpr(n_lo):
                # Int64 row base (rows*N >= 2^31 would wrap int32); N static, so
                # the multiply strength-reduces.
                cute.autovec_copy(
                    _row_vec(mX, cutlass.Int64(rr) * N64 + col0, vec), frag
                )
                xf = frag.load().to(acc_dtype)
                for v in cutlass.range_constexpr(vec):
                    accs[v] = reduce_fn(accs[v], xf[v], rr, col0 + v < N)
                rr = rr + row_stride
            n_full = rows // row_stride
            for _ in cutlass.range(n_full - n_lo):
                cute.autovec_copy(
                    _row_vec(mX, cutlass.Int64(rr) * N64 + col0, vec), frag
                )
                xf = frag.load().to(acc_dtype)
                for v in cutlass.range_constexpr(vec):
                    accs[v] = reduce_fn(accs[v], xf[v], rr, col0 + v < N)
                rr = rr + row_stride
            ok = rr < rows
            cute.autovec_copy(
                _row_vec(mX, cutlass.Int64(rr if ok else Int32(0)) * N64 + col0, vec),
                frag,
            )
            xf = frag.load().to(acc_dtype)
            for v in cutlass.range_constexpr(vec):
                accs[v] = reduce_fn(accs[v], xf[v], rr, ok and (col0 + v < N))
        else:
            # Combine pre-reduced partials: nfields buffers, each (rows, N).
            # Stage 2's rows = grid_y is tiny; single guarded dynamic loop.
            combine_fn = trait.combine
            frags = [
                cute.make_rmem_tensor(cute.make_layout(vec), mIns[f].element_type)
                for f in range(nf)
            ]
            rr = gty0
            n_full = rows // row_stride
            for _ in cutlass.range(n_full, unroll=4):
                # Int64 base (rows*N may exceed 2^31); inlined -- a name assigned
                # only inside a dynamic loop trips the IR flattener.
                for f in cutlass.range_constexpr(nf):
                    cute.autovec_copy(
                        _row_vec(mIns[f], cutlass.Int64(rr) * N64 + col0, vec),
                        frags[f],
                    )
                for v in cutlass.range_constexpr(vec):
                    part = tuple(frags[f][v] for f in range(nf))
                    merged = combine_fn(accs[v], part)
                    valid = col0 + v < N
                    accs[v] = tuple(
                        (merged[f] if valid else accs[v][f]) for f in range(nf)
                    )
                rr = rr + row_stride
            ok = rr < rows
            rb64 = cutlass.Int64(rr if ok else Int32(0)) * N64 + col0
            for f in cutlass.range_constexpr(nf):
                cute.autovec_copy(_row_vec(mIns[f], rb64, vec), frags[f])
            for v in cutlass.range_constexpr(vec):
                part = tuple(frags[f][v] for f in range(nf))
                merged = combine_fn(accs[v], part)
                valid = ok and (col0 + v < N)
                accs[v] = tuple((merged[f] if valid else accs[v][f]) for f in range(nf))

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
        # project (RUNTIME true_m divisor) one value per column; not-final -> store
        # RAW per-field accumulators to parts[f][by, col] for the stage-2 combine.
        if const_expr(self.final):
            for v in cutlass.range_constexpr(vec):
                col = col0 + v
                res = mOuts[0].element_type(trait.project(accs[v], acc_dtype(true_m)))
                if ty == 0 and col < N:
                    mOuts[0][col] = res
        else:
            for v in cutlass.range_constexpr(vec):
                col = col0 + v
                vals = [trait.fdtypes[f](accs[v][f]) for f in range(nf)]
                if ty == 0 and col < N:
                    for f in cutlass.range_constexpr(nf):
                        mOuts[f][cutlass.Int64(by) * N64 + col] = vals[f]


@cute.jit
def _row_vec(mX, base, vec: cutlass.Constexpr):
    # A vec-wide contiguous slice of the flat input starting at `base`, viewed as
    # a (vec,) tensor for autovec_copy.
    return cute.make_tensor(mX.iterator + base, cute.make_layout(vec))


_CACHE = {}


def _aligned(t, align, read_only=False):
    # Static wrap at the given alignment (K2 stage-2 partials: all extents are
    # (grid_y, N)-derived and already key the plan).
    w = _L.ReadOnlyTensorWrapper(t) if read_only else t
    ct = cute.runtime.from_dlpack(w, assumed_align=align, enable_tvm_ffi=True)
    ct.element_type = _L.torch2cute[t.dtype]
    return ct


def _aligned_dyn(t, align, read_only=False):
    # 1D wrap with the length DYNAMIC (K2 kernels are structural: M/N are runtime
    # args, so the operand extents must not bake either). K2 addresses the flat
    # tensor by explicit Int64 offsets (_row_vec), so only the extent matters.
    # enable_tvm_ffi: fast torch->tvm-ffi C exchange (~0.8us vs ~3.6us capsule).
    # read_only wraps an INPUT so a COW input exports without materializing (launch._ro).
    w = _L.ReadOnlyTensorWrapper(t) if read_only else t
    ct = cute.runtime.from_dlpack(w, assumed_align=align, enable_tvm_ffi=True)
    ct.element_type = _L.torch2cute[t.dtype]
    ct.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
    return ct


# K2 outputs are wrapped STATIC (plain _cute) and their length keys the plan:
# measured (B200, 4096x16384 f32): with BOTH operands dynamic-extent the kernel
# runs ~25% slower than with at most ONE dynamic operand (185us vs 147us, same
# runtime-N addressing in both) -- so the INPUT (whose length varies with M, the
# high-churn batch axis) stays dynamic and the output (N,) goes static. Kernel
# count is then O(#distinct N x vec-class) instead of O(#(M, N)).


def _choose_config(M, N, hw, elem_bits=32, nfields: int = 1) -> "_ColConfig":
    # Launch config for the column reduction, as a pure function of (M, N, hw, dtype).
    #
    # NO-REGRESSION baseline. A dense landscape sweep (characterize_k2.py, f32/f16 x
    # 11 shapes) showed there's real headroom (up to +0.3x on some shapes) BUT no
    # single closed-form rule for block_x/block_y/grid_y captures the surface without
    # REGRESSING other shape classes -- every scalar formula trades square/wide against
    # tall-narrow. So those stay the proven heuristic and the headroom is left to a
    # future AUTOTUNER (measures per shape-key -> cannot regress).
    #
    # vec_bits IS chosen here, dtype-aware: a graph-mode landscape (6 shapes) showed
    # narrow dtypes (bf16/fp16) want 64-bit vectors on EVERY shape (up to ~2x vs the old
    # hardcoded 128 -- this was a real loss the benchmark exposed), while fp32 keeps 128
    # (64 helps most fp32 shapes but regresses wide-N, so 128 is the fp32 no-regression
    # anchor). See _VEC_BITS_NARROW/_WIDE.
    #
    # nfields: the sweep confirmed wide-accumulator traits (var nf=3) prefer a different
    # grid_y/block_y, but the shift is shape-dependent with no no-regression closed form,
    # so nfields is accepted for parity and left to the autotuner.
    #
    # block_x/block_y fixed at the validated defaults (the _ColConfig defaults); grid_y
    # from _choose_grid_y: split the M (reduced) axis only when the column-blocks
    # underfill the device, to ~_FILL_TARGET_MULT*sm_count blocks, capped by available
    # M. sm_count is read from hw so the fill target tracks the GPU.
    vec_bits = _VEC_BITS_NARROW if elem_bits <= _NARROW_MAX_BITS else _VEC_BITS_WIDE
    vec = max(vec_bits // elem_bits, 1)  # elems/vector for the grid_x fill estimate
    grid_x = -(-N // (_DEFAULT_BLOCK_X * vec))
    gy = _choose_grid_y(M, grid_x, _DEFAULT_BLOCK_Y, hw.sm_count)
    return _ColConfig(grid_y=gy, vec_bits=vec_bits)


# Cap on the constexpr phase-A wave count (n_lo = m_floor // row_stride): the
# fold to the floor is FULLY UNROLLED at compile time (that is where the phase-A
# win comes from), so an uncapped n_lo scales compile time/kernel size with M on
# the grid_y==1 path. 512 waves ~ a few-KB loop body; beyond it phase A stops
# and the dynamic phase B carries the rest (those huge-M single-launch shapes are
# bandwidth-bound anyway). The M-split path is already bounded by _MIN_VPT.
_MAX_UNROLLED_WAVES = 512
# Phase A applies only when the column-block count is at most this (the
# tall-narrow class; see the m_floor gate in reduce_col).
_PHASE_A_MAX_GRID_X = 16


def _m_bucket_floor(M):
    # Half-octave bucket FLOOR for M (rungs 2^k and 3*2^(k-1), floor 64): the
    # kernel bakes m_floor unguarded waves, so every M in [floor, next rung)
    # shares one compiled kernel and at most ~33% of the fold runs in the guarded
    # dynamic phase. Below 64 rows the fold is trivially short -- m_floor=0.
    if M < 64:
        return 0
    p = 1 << (M.bit_length() - 1)  # largest 2^k <= M
    half = (3 * p) // 2
    return half if half <= M else p


def _choose_grid_y(M, grid_x, block_y, sm, min_vpt=_MIN_VPT):
    # ctas_per_output for the M (reduced) axis. Split M ONLY when the column-blocks
    # alone leave the device underfilled (splitting adds a gmem partial round-trip, so
    # it only pays when it buys occupancy). Near a full wave don't split. Otherwise
    # target the fill goal; cap so each thread-row still folds >= min_vpt rows. All
    # thresholds are the named module constants above; sm is hw.sm_count (portable).
    nw_num, nw_den = _NEAR_WAVE_FRAC
    if grid_x >= (nw_num * sm) // nw_den:
        return 1
    gy = max(1, -(-(_FILL_TARGET_MULT * sm) // max(grid_x, 1)))
    return max(1, min(gy, max(1, M // (block_y * min_vpt))))


def reduce_col(
    trait,
    trait_key,
    x,
    out_dtype,
    block_x=None,
    block_y=None,
    grid_y=None,
    vec_bits=None,
):
    # Vectorized column reduction (reduce dim 0). x: (M, N) contiguous.
    # block_x/block_y/grid_y/vec_bits default to the HW+dtype heuristic (_choose_config);
    # pass explicit values to override (autotuner / retune). vec_bits is the load/store
    # vector width in bits (heuristic picks 64 for narrow dtypes, 128 for fp32); gcd with
    # N keeps it a legal divisor for ragged N.
    assert x.dim() == 2 and x.is_cuda and x.stride(-1) == 1  # noqa: S101
    M, N = x.shape
    # The config (and the baked m_floor/grid_y) derive from M's HALF-OCTAVE BUCKET
    # floor, not exact M: every M in a bucket then shares one kernel per (N, path),
    # the first m_floor waves of the fold run unguarded/compile-time (the perf fix
    # for the M-split shapes), and only the <= 25% tail is the guarded dynamic
    # loop. Explicit grid_y (autotuner) bypasses the bucket (m_floor=0, exact key).
    # Config from the bucket floor (bucket-stable: every M in the bucket picks
    # the same knobs, so the baked grid_y/m_floor never flip inside a bucket).
    bucket_floor = _m_bucket_floor(M) if grid_y is None else 0
    cfg_m = bucket_floor if bucket_floor > 0 else M
    cfg = _choose_config(
        cfg_m, N, _hw.caps(x.device), x.element_size() * 8, trait.nfields
    )
    block_x = cfg.block_x if block_x is None else block_x
    block_y = cfg.block_y if block_y is None else block_y
    grid_y = cfg.grid_y if grid_y is None else grid_y
    vec_bits = cfg.vec_bits if vec_bits is None else vec_bits
    # Bake the bucket floor ONLY for the TALL-NARROW M-split class (few column
    # blocks, deep fold): those stripes are latency-bound and gain ~20% from the
    # unrolled phase A (B200 65536x1024, grid_x=4: 46 vs 56us). Wide-N shapes with
    # the SAME byte count (4096x16384, grid_x=64) measured ~9% SLOWER with phase A
    # -- their many column blocks already cover latency -- as did the grid_y==1
    # long-fold path; both keep the plain dynamic loop. Clamped so phase A's full
    # unroll stays bounded; row_stride-scaled so the WAVE count is what's capped.
    vec = math.gcd(N, vec_bits // (x.element_size() * 8))
    grid_x = -(-N // (block_x * max(vec, 1)))
    tall_narrow = grid_y > 1 and grid_x <= _PHASE_A_MAX_GRID_X
    m_floor = (
        min(bucket_floor, _MAX_UNROLLED_WAVES * grid_y * block_y) if tall_narrow else 0
    )
    out = torch.empty(N, device=x.device, dtype=out_dtype)
    xf = x.reshape(-1)
    align = 16 if (N % vec == 0) else x.element_size()

    # Compile keys are STRUCTURAL only -- (vec-class, block shape, path); M / N /
    # grid_y are runtime launch args (ColReduce.geom_args), so every (M, N) with the
    # same vec class shares ONE kernel per path. The 1D operand wraps mark the
    # length dynamic for the same reason. grid_y stays a HOST decision (it sizes
    # the partial buffers and picks the 1- vs 2-stage path).
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
            m_floor=m_floor,
        )
        xin = _aligned_dyn(xf, align, read_only=True)
        # N in the key: the output wrap is static (see the note above _aligned).
        # m_floor is in cache_sig, so M-buckets key distinct kernels.
        key = ("col", trait_key, x.dtype, out_dtype, N) + op.cache_sig
        fn = cached_plan(
            _CACHE,
            key,
            lambda: _compile(op, [xin], [_cute(out)], *op.geom_args(), _stream()),
            op=f"aten::{key[1]}",
        )
        fn([xin], [_cute(out)], *op.geom_args(), _stream())
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
        m_floor=m_floor,
    )
    xin = _aligned_dyn(xf, align, read_only=True)
    # (N, grid_y) key the static partial wraps (grid_y*N,); the input stays the
    # single dynamic operand so M never recompiles.
    k1 = ("col-s1", trait_key, x.dtype, N, grid_y) + op1.cache_sig
    f1 = cached_plan(
        _CACHE,
        k1,
        lambda: _compile(
            op1, [xin], [_cute(p) for p in parts], *op1.geom_args(), _stream()
        ),
        op=f"aten::{k1[1]}",
    )
    f1([xin], [_cute(p) for p in parts], *op1.geom_args(), _stream())

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
    # Stage 2 is all-static: every extent is (grid_y*N)- or N-derived, M appears
    # only in the runtime true_m divisor. Cheap kernel (folds grid_y partials).
    pin = [_aligned(p, palign, read_only=True) for p in parts]
    k2 = ("col-s2", trait_key, out_dtype, N, grid_y) + op2.cache_sig
    f2 = cached_plan(
        _CACHE,
        k2,
        lambda: _compile(op2, pin, [_cute(out)], *op2.geom_args(), _stream()),
        op=f"aten::{k2[1]}",
    )
    f2(pin, [_cute(out)], *op2.geom_args(), _stream())
    return out
