# Copyright (c) 2026, Tri Dao.
"""Device-side core for block-scaled output quantization (SFD).

Shared by kernels that emit a (values, scale-factor) pair consumed by a
downstream blockscaled GEMM: the quantized-output GEMM epilogue
(quack.epilogue.quantize_out.BlockScaleFactorStore) and the fused
RMSNorm/LayerNorm quantized forward.

Semantics contract — bit-exact with cuBLAS and the CUTLASS C++
Sm1xxBlockScaleFactorRowStore:

  sf      = cvt(amax / value_dtype_max)     f32->e8m0 rounds toward +inf,
                                            f32->e4m3 rounds to nearest even
  rescale = min(rcp(dequant(sf)), FLT_MAX)  exact for e8m0 (byte trick below)
  q       = cvt_value_dtype(y * rescale)    RNE satfinite

The e8m0 reciprocal is exact without MUFU.RCP: negating the biased exponent
byte gives rcp(2^(b-127)) = 2^((254-b)-127); NaN (0xFF) wraps back to 0xFF
and propagates. Rescaling by the reciprocal of the *quantized* scale (not the
raw amax/max ratio) keeps the stored values consistent with the stored SF
byte.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.base_dsl.arch import Arch

import torch._vendor.quack.utils as utils
import torch._vendor.quack.copy_utils as copy_utils

# Max representable value of the quantized value dtype (the "MAX" in
# scale = amax / MAX). Mirrors cutlass::platform::numeric_limits<T>::max().
QUANT_DTYPE_MAX = {
    cutlass.Float8E4M3FN: 448.0,
    cutlass.Float8E5M2: 57344.0,
    cutlass.Float4E2M1FN: 6.0,
}

FLT_MAX = 3.40282346638528859812e38


# sf_vec_size implied by a (value_dtype, sf_dtype) pair: e8m0 scales (mx
# formats) cover 32 values, e4m3 scales (nvfp4) cover 16.
def sf_vec_size_for(sf_dtype) -> int:
    return 32 if sf_dtype is cutlass.Float8E8M0FNU else 16


@cute.jit
def quantize_sf_slots(tAmax_flt, tSFD_flt, tScale_flt, value_dtype, norm_const=None):
    """amax slots -> SF bytes + f32 rescale factors, one slot per SF vector.

    tAmax_flt:  f32 per-slot absolute maxima (already reduced over the full
                SF vector, including any cross-lane combine).
    tSFD_flt:   SF-dtype (e8m0 or e4m3) output slots for the byte to store.
    tScale_flt: f32 output slots; multiply each value by its slot's factor
                before the ordinary f32 -> value_dtype conversion.
    norm_const: optional extra factor folded into both the SF and the rescale
                (per-tensor scale for nvfp4); None for mx formats.

    All three are flat filtered views with matching slot order. Scalar setitem
    throughout: TensorSSA load/store on filtered views whose layout degenerates
    to all-zero strides is silently dropped by the DSL.
    """
    rcp_dmax = 1.0 / QUANT_DTYPE_MAX[value_dtype]
    if const_expr(norm_const is not None):
        norm_scaled = norm_const * rcp_dmax
    else:
        norm_scaled = Float32(rcp_dmax)
    sf_dtype = tSFD_flt.element_type
    is_e8m0 = const_expr(sf_dtype is cutlass.Float8E8M0FNU)
    n_slots = cute.size(tSFD_flt)
    if const_expr(is_e8m0):
        tRcpB = cute.make_rmem_tensor(cute.make_layout(n_slots), cutlass.Uint8)
        tRcp_e8 = cute.recast_tensor(tRcpB, cutlass.Float8E8M0FNU)
        tSFD_u8 = cute.recast_tensor(tSFD_flt, cutlass.Uint8)
    for vi in cutlass.range_constexpr(n_slots):
        tSFD_flt[vi] = sf_dtype(tAmax_flt[vi] * norm_scaled)
    for vi in cutlass.range_constexpr(n_slots):
        if const_expr(is_e8m0):
            tRcpB[vi] = cutlass.Uint8((254 - tSFD_u8[vi]) & 0xFF)
            rescale = tRcp_e8[vi].to(Float32)
        else:
            rescale = cute.arch.rcp_approx(tSFD_flt[vi].to(Float32))
        if const_expr(norm_const is not None):
            rescale = rescale * norm_const
        tScale_flt[vi] = cute.arch.fmin(rescale, FLT_MAX)


@cute.jit
def quantize_chunk_rowwise(tYf, tQc, sf_dtype, lane_span: cutlass.Constexpr[int]):
    """Quantize one row-wise chunk of ``vecsize`` fp32 values into tQc.

    amax over the fragment, butterfly-combined across the ``lane_span``
    adjacent lanes sharing the SF vector (adjacent lanes hold adjacent chunks;
    the caller guarantees lane-aligned groups), then quantize_sf_slots on the
    single slot and the rescaled fp8 conversion. Every lane of the vector
    holds the returned SF value; the lane owning the vector's first chunk
    stores it. Shared by the full-fragment and chunked row-quant paths so the
    two cannot drift.
    """
    vecsize = const_expr(cute.size(tYf))
    # Off SM100: max.xorsign.abs fold — maximum magnitude with XORed signs,
    # no per-element absf, one sign-clearing abs after the full fold (same
    # pattern as the VecReduce max_abs epilogue). On SM100 keep the plain
    # fmax(acc, |x|) chain: ptxas fuses it into 3-input FMNMX3.ABS (abs is an
    # operand modifier there, and FMNMX3 is SM100-only), which the two-input
    # xorsign form would defeat.
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    xorsign = const_expr(not arch.is_family_of(Arch.sm_100f))
    amax = Float32(0.0)
    for v in cutlass.range_constexpr(vecsize):
        if const_expr(xorsign):
            amax = cute.arch.fmax(amax, tYf[v], abs=True)
        else:
            amax = cute.arch.fmax(amax, cute.math.absf(tYf[v]))
    if const_expr(lane_span > 1):
        step = lane_span // 2
        while step > 0:
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=step), abs=xorsign)
            step //= 2
    if const_expr(xorsign):
        amax = cute.math.abs(amax)
    tAmax1 = cute.make_rmem_tensor(cute.make_layout(1), Float32)
    tSFD1 = cute.make_rmem_tensor(cute.make_layout(1), sf_dtype)
    tScale1 = cute.make_rmem_tensor(cute.make_layout(1), Float32)
    tAmax1[0] = amax
    quantize_sf_slots(tAmax1, tSFD1, tScale1, tQc.element_type)
    for v in cutlass.range_constexpr(vecsize):
        tQc[v] = tQc.element_type(tYf[v] * tScale1[0])
    return tSFD1[0]


@cute.jit
def quantize_output_rowwise(y, gQ, mSFD, tXcX, tXpX, thr_copy_X, copy, row, shape, is_even_N):
    """Emit the fp8 quantized copy of a row-major output plus its block scale factors.

    Quantizes the fp32 ``y`` (not the rounded low-precision store), matching the
    quantized-output GEMM epilogue which quantizes the fp32 accumulator.
    Each lane owns contiguous ``vecsize``-element chunks of the row; an SF
    vector of ``sf_vec`` columns spans ``sf_vec // vecsize`` adjacent lanes,
    whose partial amaxes are butterfly-combined so the rescale stays
    lane-local. SF bytes go through the logical (M_pad, N_pad) view whose
    intra-vector mode has stride 0; the lane holding a vector's first chunk
    stores the byte. Shared by the RMSNorm fwd (y) and bwd (dx) kernels.
    """
    tXgQ = thr_copy_X.partition_D(gQ)
    tXrQ = cute.make_rmem_tensor_like(tXgQ)
    sf_vec = const_expr(sf_vec_size_for(mSFD.element_type))
    vecsize = const_expr(cute.size(tXgQ, mode=[0]))
    nb = const_expr(cute.size(tXgQ, mode=[2]))
    # fp32 output values with OOB lanes zeroed: they must not poison the
    # amax (predicated weight/bias loads leave OOB register lanes
    # uninitialized, so 0*w or +b can be non-finite there).
    tXrYf = cute.make_rmem_tensor_like(tXrQ, Float32)
    tXrYf.store(y)
    if const_expr(not is_even_N):
        utils.fill_oob(tXrYf, tXpX, Float32.zero)
    # Adjacent lanes hold adjacent chunks of the same SF vector (thr_layout
    # is row-fastest), and threads_per_row % lane_span == 0 keeps the
    # groups lane-aligned, so the helper's butterfly combine works.
    lane_span = const_expr(sf_vec // vecsize)
    tSFD = cute.make_rmem_tensor(cute.make_layout(nb), mSFD.element_type)
    for b in cutlass.range_constexpr(nb):
        tSFD[b] = quantize_chunk_rowwise(
            tXrYf[None, 0, b], tXrQ[None, 0, b], mSFD.element_type, lane_span
        )
    # SF byte flush: one predicated STG.U8 per vector from the lane holding
    # its first chunk. Do NOT "optimize" this into shuffle-gathered 32-bit
    # packed stores (the blocked atom does put 4 k-slots at consecutive
    # 4-byte-aligned addresses, and the bytes live in lanes
    # lane0 + {0,1,2,3} * lane_span): measured on B300 2026-07-10 it was a
    # consistent 3-14% end-to-end regression — the 3 warp shuffles + OR
    # chain per chunk run on EVERY lane, while these byte stores are
    # predicated to 1-in-lane_span lanes and carry ~3% of the q traffic.
    # Instruction issue, not SF store transactions, is what binds here.
    if row < shape[0]:
        copy(tXrQ, tXgQ)
        for b in cutlass.range_constexpr(nb):
            col = tXcX[(0, 0, b)][1]
            if col % sf_vec == 0:
                if col < shape[1]:
                    mSFD[row, col] = tSFD[b]


@cute.jit
def quantize_output_colwise(gQC, mSFC, sX, cX, tidx, shape, vecsize):
    """Col-wise (dim-M) quantization from a restaged 32-row smem slab.

    Each participating thread owns a full 32-row x ``vecsize``-column strip,
    so the per-column amax, SF byte, and rescale are thread-local — no
    shuffles, no cross-thread exchange. Dynamic row loops + immediate
    per-row stores: unrolled loops let the compiler keep every row's
    fragment in flight (a full 32-row q fragment alone is 64 registers),
    defeating the chunked row phase's register savings; a single-sweep
    variant caching the strip between amax and rescale costs the same
    512 B/thread and cannot index registers across dynamic iterations.
    Output values keep the (M, N) orientation of the row path; only the
    scales come from column maxima. SF bytes go through the
    transposed-atom logical (N_pad, M_pad) view. Shared by the RMSNorm
    fwd (y) and bwd (dx) kernels.
    """
    sf_vec_c = const_expr(cute.size(sX, mode=[0]))  # rows per CTA == 32
    tiler_n = const_expr(cute.size(sX, mode=[1]))
    t_active = const_expr(tiler_n // vecsize)
    tiled_copy_col = copy_utils.tiled_copy_2d(gQC.element_type, t_active, t_active, vecsize)
    if tidx < t_active:
        thr_col = tiled_copy_col.get_slice(tidx)
        tCsY = thr_col.partition_S(sX)  # ((1, vecsize), sf_vec_c, 1)
        tCgQC = thr_col.partition_D(gQC)
        tCcX = thr_col.partition_S(cX)[(0, None), None, None]
        tAmaxC = cute.make_rmem_tensor(cute.make_layout(vecsize), Float32)
        tAmaxC.fill(0.0)
        for r in cutlass.range(sf_vec_c, unroll=2):
            vals = tCsY[None, r, 0]
            # TODO: use the same FMNMX3.ABS or FMNMX XORSIGN trick
            for v in cutlass.range_constexpr(vecsize):
                tAmaxC[v] = cute.arch.fmax(tAmaxC[v], cute.math.absf(vals[v].to(Float32)))
        tSFC = cute.make_rmem_tensor(cute.make_layout(vecsize), mSFC.element_type)
        tScaleC = cute.make_rmem_tensor(cute.make_layout(vecsize), Float32)
        quantize_sf_slots(tAmaxC, tSFC, tScaleC, gQC.element_type)
        row0 = tCcX[(0, 0, 0)][0]
        col0 = tCcX[(0, 0, 0)][1]
        for r in cutlass.range(sf_vec_c, unroll=2):
            vals = tCsY[None, r, 0]
            qrow = cute.make_rmem_tensor_like(tCgQC[None, 0, 0])
            for v in cutlass.range_constexpr(vecsize):
                qrow[v] = gQC.element_type(vals[v].to(Float32) * tScaleC[v])
            if tCcX[(0, r, 0)][0] < shape[0]:
                if col0 < shape[1]:
                    cute.autovec_copy(qrow, tCgQC[None, r, 0])
        for v in cutlass.range_constexpr(vecsize):
            if col0 + v < shape[1]:
                mSFC[col0 + v, row0] = tSFC[v]
