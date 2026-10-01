"""Shared helpers for handwritten CuTe DSL quantization kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import dsl_user_op, T


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


@cute.jit
def _store_scale_bytes_as_uint(
    mScaleLogical: cute.Tensor,
    rScale: cute.Tensor,
    row: cutlass.Int32,
    col: cutlass.Int32,
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
    row: cutlass.Int32,
    scale_col: cutlass.Int32,
    row_stride: cutlass.Int32,
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

    qdata = (values * reciprocal).to(cutlass.Float8E4M3FN)
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
