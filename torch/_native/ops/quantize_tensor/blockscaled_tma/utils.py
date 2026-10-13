"""Shared helpers for handwritten CuTe DSL quantization kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import dsl_user_op, T

from .blockscaled_tma_config import RoundingVariant


_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


def _ceil_div(
    num: int | cutlass.Int32 | cutlass.Int64,
    den: int | cutlass.Int32 | cutlass.Int64 | cutlass.Constexpr,
) -> int | cutlass.Int32 | cutlass.Int64:
    # Avoid overflowing a fixed-width numerator close to its maximum value.
    return num // den + (num % den != 0)


@dsl_user_op
def view_as(
    x: cutlass.Numeric,
    dtype: type[cutlass.Numeric],
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """Bitcast one scalar to another scalar of equal width."""
    if type(x).width != dtype.width:
        raise AssertionError(
            f"expected equal bit widths, got {type(x).width} and {dtype.width}"
        )
    # bitcast wants a signed IR type even for unsigned CUTLASS types.
    dst_type = (
        T.i(dtype.width)
        if ir.IntegerType.isinstance(dtype.mlir_type)
        else dtype.mlir_type
    )
    return dtype(arith.bitcast(dst_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def unpack(
    x: cutlass.Numeric,
    dtype: type[cutlass.Numeric],
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[cutlass.Numeric, ...]:
    """Unpack an integer carrier into a tuple of scalar values."""
    x = cute.typing.as_numeric(x)
    carrier_dtype = type(x)
    if not ir.IntegerType.isinstance(carrier_dtype.mlir_type):
        raise AssertionError(f"expected an integer carrier, got {carrier_dtype}")
    if carrier_dtype.width % dtype.width != 0:
        raise AssertionError(
            f"expected carrier width divisible by {dtype.width}, got {carrier_dtype.width}"
        )
    num_lanes = carrier_dtype.width // dtype.width
    # Integer vector lanes avoid a compiler crash with vector<N x FP8> (NVIDIA/cutlass#3342).
    lanes = llvm.bitcast(
        T.vector(num_lanes, T.i(dtype.width)),
        x.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return tuple(
        view_as(
            cute.typing.as_numeric(
                vector.extract(
                    lanes,
                    dynamic_position=[],
                    static_position=[i],
                    loc=loc,
                    ip=ip,
                )
            ),
            dtype,
            loc=loc,
            ip=ip,
        )
        for i in range(num_lanes)
    )


@dsl_user_op
def pack(
    *values: cutlass.Numeric,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """Pack same-typed scalar values into an integer carrier."""
    if len(values) == 0:
        raise AssertionError("expected at least one value, got 0")
    lane_dtype = type(values[0])
    if any(type(value) is not lane_dtype for value in values):
        raise AssertionError(
            f"expected all values of type {lane_dtype}, got {tuple(type(value) for value in values)}"
        )
    lane_type = T.i(lane_dtype.width)
    lanes = vector.from_elements(
        T.vector(len(values), lane_type),
        tuple(
            arith.bitcast(
                lane_type,
                value.ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
            for value in values
        ),
        loc=loc,
        ip=ip,
    )
    packed_width = len(values) * lane_dtype.width
    packed = llvm.bitcast(T.i(packed_width), lanes, loc=loc, ip=ip)
    return cute.typing.as_numeric(packed)


@dsl_user_op
def _cvt_f32_to_ue8m0(
    x: cutlass.Float32,
    *,
    rounding_mode: nvvm.FPRoundingMode,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float8E8M0FNU:
    """Convert x to a single E8M0 value without saturation (high input 0.0)."""
    packed = nvvm.cvt_packfloat_f32(
        cutlass.Float32(0.0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(x).ir_value(loc=loc, ip=ip),
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        rnd=rounding_mode,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc,
        ip=ip,
    )
    return unpack(
        cutlass.Int32(packed),
        cutlass.Float8E8M0FNU,
        loc=loc,
        ip=ip,
    )[0]


@dsl_user_op
def _cvt_ue8m0_to_f32(
    x: cutlass.Float8E8M0FNU,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    """Convert one E8M0 value to FP32 through the supported BF16 path."""
    x_e8m0x2 = pack(x, cutlass.Float8E8M0FNU(0), loc=loc, ip=ip)
    x_u32 = llvm.zext(T.i32(), x_e8m0x2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    bf16x2_bits = nvvm.cvt_packfloat(
        x_u32,
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        nvvm.CVTPackFloatKind.BF16x2,
        rnd=nvvm.FPRoundingMode.RN,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc,
        ip=ip,
    )
    low_bf16 = unpack(
        cutlass.Int32(bf16x2_bits),
        cutlass.BFloat16,
        loc=loc,
        ip=ip,
    )[0]
    return low_bf16.to(cutlass.Float32)


@cute.jit
def _reciprocal_scale(
    scale_e8m0: cutlass.Float8E8M0FNU,
) -> cutlass.Float32:
    scale_biased = view_as(scale_e8m0, cutlass.Uint8)
    reciprocal_biased = cutlass.Uint8(254) - scale_biased
    return _cvt_ue8m0_to_f32(view_as(reciprocal_biased, cutlass.Float8E8M0FNU))


@cute.jit
def _e8m0(amax: cutlass.Float32) -> tuple[cutlass.Float32, cutlass.Uint8]:
    """Return the reciprocal FP32 scale and RCEIL E8M0 scale byte for an amax."""
    descale = amax * cutlass.Float32(1.0 / 448.0)
    scale_e8m0 = _cvt_f32_to_ue8m0(descale, rounding_mode=nvvm.FPRoundingMode.RP)
    rcp = _reciprocal_scale(scale_e8m0)
    return rcp, view_as(scale_e8m0, cutlass.Uint8)


@dsl_user_op
def _cvt_rs_satfinite_e4m3x4_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    rbits: cutlass.Uint32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    """Stochastically round four FP32 values to four packed E4M3 bytes."""
    # PTX packs its first source into the high byte; reverse them for little-endian order.
    args = [
        cutlass.Float32(value).ir_value(loc=loc, ip=ip) for value in (v3, v2, v1, v0)
    ]
    args.append(cutlass.Uint32(rbits).ir_value(loc=loc, ip=ip))
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
            "=r,f,f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _philox_4x32(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10, matching CUDA's counter-based generator."""
    for _ in cutlass.range_constexpr(10):
        p0 = cutlass.Uint64(c0) * cutlass.Uint64(_PHILOX_M0)
        p1 = cutlass.Uint64(c2) * cutlass.Uint64(_PHILOX_M1)
        hi0 = cutlass.Uint32(p0 >> 32)
        lo0 = cutlass.Uint32(p0 & cutlass.Uint64(0xFFFFFFFF))
        hi1 = cutlass.Uint32(p1 >> 32)
        lo1 = cutlass.Uint32(p1 & cutlass.Uint64(0xFFFFFFFF))
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = k0 + cutlass.Uint32(_PHILOX_W0)
        k1 = k1 + cutlass.Uint32(_PHILOX_W1)
    return c0, c1, c2, c3


@cute.jit
def _philox_counter_words(
    rounding_variant: cutlass.Constexpr,
    operation_block: cutlass.Uint64,
    logical_block: cutlass.Uint64,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    if cutlass.const_expr(rounding_variant == RoundingVariant.STATELESS_SR):
        counter = operation_block + logical_block
        subsequence = cutlass.Uint64(0)
    else:
        counter = operation_block
        subsequence = logical_block
    return (
        cutlass.Uint32(counter & cutlass.Uint64(0xFFFFFFFF)),
        cutlass.Uint32(counter >> 32),
        cutlass.Uint32(subsequence & cutlass.Uint64(0xFFFFFFFF)),
        cutlass.Uint32(subsequence >> 32),
    )


@cute.jit
def _mxfp8_quantize_stochastic_x32(
    values: cute.TensorSSA,
    reciprocal: cutlass.Float32,
    operation_block: cutlass.Uint64,
    logical_block_start: cutlass.Uint64,
    philox_k0: cutlass.Uint32,
    philox_k1: cutlass.Uint32,
    rounding_variant: cutlass.Constexpr,
) -> cute.TensorSSA:
    scaled = values * reciprocal
    qwords = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        c0, c1, c2, c3 = _philox_counter_words(
            rounding_variant,
            operation_block,
            logical_block_start + cutlass.Uint64(half),
        )
        r0, r1, r2, r3 = _philox_4x32(c0, c1, c2, c3, philox_k0, philox_k1)
        value = half * 16
        word = half * 4
        qwords[word + 0] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 0],
            scaled[value + 1],
            scaled[value + 2],
            scaled[value + 3],
            r0,
        )
        qwords[word + 1] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 4],
            scaled[value + 5],
            scaled[value + 6],
            scaled[value + 7],
            r1,
        )
        qwords[word + 2] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 8],
            scaled[value + 9],
            scaled[value + 10],
            scaled[value + 11],
            r2,
        )
        qwords[word + 3] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 12],
            scaled[value + 13],
            scaled[value + 14],
            scaled[value + 15],
            r3,
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Float8E4M3FN).load()


@cute.jit
def _load_philox_key_and_counter(
    mSeed: cute.Tensor,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint64]:
    frgKey = cute.make_rmem_tensor(cute.make_layout(2), mSeed.element_type)
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mSeed.element_type),
        mSeed,
        frgKey,
    )
    key64 = cute.recast_tensor(frgKey, dtype=cutlass.Uint64)
    return (
        cutlass.Uint32(key64[0] & cutlass.Uint64(0xFFFFFFFF)),
        cutlass.Uint32(key64[0] >> 32),
        key64[1],
    )


@cute.jit
def _stateful_philox_seed_and_counter(
    seed64: cutlass.Uint64,
    offset_words64: cutlass.Uint64,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint64]:
    return (
        cutlass.Uint32(seed64 & cutlass.Uint64(0xFFFFFFFF)),
        cutlass.Uint32(seed64 >> 32),
        offset_words64 >> 2,
    )


@cute.jit
def _load_stateful_philox_device_state(
    mSeed: cute.Tensor,
    mOffset: cute.Tensor,
    intragraph_offset_words: cutlass.Int64,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint64]:
    frgSeed = cute.make_rmem_tensor(cute.make_layout(1), mSeed.element_type)
    frgOffset = cute.make_rmem_tensor(cute.make_layout(1), mOffset.element_type)
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mSeed.element_type),
        mSeed,
        frgSeed,
    )
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOffset.element_type),
        mOffset,
        frgOffset,
    )
    seed64 = cute.recast_tensor(frgSeed, dtype=cutlass.Uint64)[0]
    offset_words64 = cute.recast_tensor(frgOffset, dtype=cutlass.Uint64)[0]
    return _stateful_philox_seed_and_counter(
        seed64, offset_words64 + view_as(intragraph_offset_words, cutlass.Uint64)
    )


@cute.jit
def _resolve_philox_state(
    rounding_variant: cutlass.Constexpr,
    mSeed: cute.Tensor | None,
    mOffset: cute.Tensor | None,
    seed_scalar: cutlass.Int64 | None,
    offset_words_scalar: cutlass.Int64 | None,
    intragraph_offset_words: cutlass.Int64 | None,
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint64]:
    if cutlass.const_expr(rounding_variant == RoundingVariant.STATELESS_SR):
        return _load_philox_key_and_counter(mSeed)
    if cutlass.const_expr(rounding_variant == RoundingVariant.STATEFUL_SR_CAPTURE):
        return _load_stateful_philox_device_state(
            mSeed, mOffset, intragraph_offset_words
        )
    return _stateful_philox_seed_and_counter(
        view_as(seed_scalar, cutlass.Uint64),
        view_as(offset_words_scalar, cutlass.Uint64),
    )


@cute.jit
def _store_scale_bytes_as_uint(
    mScaleLogical: cute.Tensor,
    rScale: cute.Tensor,
    row: cutlass.Int64,
    col: cutlass.Int64,
    count: cutlass.Constexpr,
) -> None:
    """Store adjacent one-byte scale values with one naturally sized integer write.

    Args:
        mScaleLogical: Global scale tensor with the logical-to-blocked layout.
        rScale: Register tensor holding the E8M0 byte values.
        row: Logical destination row.
        col: First logical destination column, aligned to ``count``.
        count: Compile-time store width in bytes; must be 1, 2, or 4.

    The explicit integer view preserves packed stores when dynamic outer extents
    prevent ``cute.copy`` from proving alignment and vectorizing the write.
    """
    flat = mScaleLogical.layout((row, col))
    # Dynamic outer extents hide this alignment from cute.copy, so use a typed packed view.
    if cutlass.const_expr(count == 1):
        mScaleLogical[(row, col)] = rScale[0]
    elif cutlass.const_expr(count == 2):
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScaleLogical.iterator, dtype=cutlass.Uint16),
            cute.make_layout(cute.size(mScaleLogical) // 2),
        )
        rScalePacked = cute.recast_tensor(rScale, dtype=cutlass.Uint16)
        mScalePacked[flat // 2] = rScalePacked[0]
    else:
        if cutlass.const_expr(count != 4):
            raise AssertionError(f"unsupported E8M0 scale store count: {count}")
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScaleLogical.iterator, dtype=cutlass.Uint32),
            cute.make_layout(cute.size(mScaleLogical) // 4),
        )
        rScalePacked = cute.recast_tensor(rScale, dtype=cutlass.Uint32)
        mScalePacked[flat // 4] = rScalePacked[0]


@cute.jit
def _store_unswizzled_scale_groups_as_uint(
    mScale: cute.Tensor,
    rScale: cute.Tensor,
    row: cutlass.Int64,
    scale_col: cutlass.Int64,
    row_stride: cutlass.Int64,
    group_count: cutlass.Constexpr,
) -> None:
    """Store adjacent bytes into a compact row-major scale tensor.

    The explicit flat offset keeps address calculation independent of the
    tensor's dynamic 2-D layout while retaining 16- or 32-bit global stores.
    Callers must ensure each destination is naturally aligned.
    """
    flat = cutlass.Int64(row) * cutlass.Int64(row_stride) + cutlass.Int64(scale_col)
    if cutlass.const_expr(group_count == 1):
        mScale[flat] = rScale[0]
    elif cutlass.const_expr(group_count == 2):
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScale.iterator, dtype=cutlass.Uint16),
            cute.make_layout(cute.size(mScale) // 2),
        )
        rScalePacked = cute.recast_tensor(rScale, dtype=cutlass.Uint16)
        mScalePacked[flat // 2] = rScalePacked[0]
    else:
        if cutlass.const_expr(group_count % 4 != 0):
            raise AssertionError(f"expected group_count % 4 == 0, got {group_count}")
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScale.iterator, dtype=cutlass.Uint32),
            cute.make_layout(cute.size(mScale) // 4),
        )
        rScalePacks = cute.tiled_divide(rScale, (4,))
        for pack_idx in cutlass.range_constexpr(group_count // 4):
            rScalePacked = cute.recast_tensor(
                rScalePacks[(None, pack_idx)], dtype=cutlass.Uint32
            )
            mScalePacked[flat // 4 + pack_idx] = rScalePacked[0]


@cute.jit
def _blockscaled_quantize_group(
    values: cute.TensorSSA,
    value_count: cutlass.Constexpr,
    is_square_scaling: cutlass.Constexpr,
    rounding_variant: cutlass.Constexpr,
    sr_operation_block: cutlass.Uint64 | None,
    sr_logical_block: cutlass.Uint64 | None,
    philox_k0: cutlass.Uint32 | None,
    philox_k1: cutlass.Uint32 | None,
):
    """Calculate one block scale and quantize the values that share it."""
    amax = cute.math.absf(values).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    if cutlass.const_expr(is_square_scaling):
        amax = cute.arch.warp_reduction(
            amax,
            op=lambda x, y: cute.arch.fmax(x, y, nan=True),
            threads_in_group=value_count,
        )

    reciprocal, scale = _e8m0(amax)

    if cutlass.const_expr(rounding_variant == RoundingVariant.RTNE):
        qdata = (values * reciprocal).to(cutlass.Float8E4M3FN)
    else:
        qdata = _mxfp8_quantize_stochastic_x32(
            values,
            reciprocal,
            sr_operation_block,
            sr_logical_block,
            philox_k0,
            philox_k1,
            rounding_variant,
        )
    return qdata, scale


@cute.jit
def _store_swizzled_scale_groups_as_uint(
    mScaleLogical: cute.Tensor,
    rScale: cute.Tensor,
    row,
    scale_col,
    group_count: cutlass.Constexpr,
):
    """Store a thread's adjacent scale bytes using their natural packed width."""
    if cutlass.const_expr(group_count in (1, 2, 4)):
        _store_scale_bytes_as_uint(mScaleLogical, rScale, row, scale_col, group_count)
    else:
        if cutlass.const_expr(group_count % 4 != 0):
            raise AssertionError(f"expected group_count % 4 == 0, got {group_count}")
        rScalePacks = cute.tiled_divide(rScale, (4,))
        for pack in cutlass.range_constexpr(group_count // 4):
            _store_scale_bytes_as_uint(
                mScaleLogical,
                rScalePacks[(None, pack)],
                row,
                scale_col + pack * 4,
                4,
            )
