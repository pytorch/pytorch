# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

# pyre-ignore-all-errors
import math
import operator
from typing import Callable, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T


# import from https://github.com/Dao-AILab/quack/blob/main/quack/utils.py
def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)"""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if cutlass.const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, cutlass.Int64)), "val must be Float32 or Int64"
    suffix = "f32" if cutlass.const_expr(isinstance(val, Float32)) else "s64"
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{'f' if cutlass.const_expr(isinstance(val, Float32)) else 'l'},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[cutlass.Int32] = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return warp_reduce(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if cutlass.const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(
            val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val
        )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax
        if cutlass.const_expr(x.dtype == Float32)
        else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = warp_reduce(
        val,
        warp_op,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert (
            cluster_n == 1 or mbar_ptr is not None
        ), "mbar_ptr must be provided for cluster reduction"
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val, warp_op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val
            )
    return val


@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    hook_fn: Optional[Callable] = None,
    phase: Optional[cutlass.Int32] = None,
    return_exp_x: bool = False,
) -> [Float32, Float32, Optional[cute.TensorSSA]]:
    assert x.dtype == Float32, "x must be of type Float32"
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n), 2)"""
    max_x = warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    log2_e = math.log2(math.e)
    exp_x = exp2f(x * log2_e - (max_x * log2_e))
    # exp_x = exp2f((x - max_x) * log2_e)
    sum_exp_x = warp_reduce(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        assert (
            cluster_n == 1 or mbar_ptr is not None
        ), "mbar_ptr must be provided for cluster reduction"
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            assert (
                reduction_buffer.element_type == cutlass.Int64
            ), "reduction_buffer must be of type cute.Int64"
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
            if cutlass.const_expr(mbar_ptr is None):
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx] = f32x2_to_i64(max_x, sum_exp_x)
                cute.arch.barrier()
                max_x_single_warp = -Float32.inf
                sum_exp_x = 0.0
                if lane_idx < warps_per_row:
                    max_x_single_warp, sum_exp_x = i64_to_f32x2(
                        reduction_buffer[row_idx, lane_idx]
                    )
                max_x_final = warp_reduce(max_x_single_warp, cute.arch.fmax)
                sum_exp_x *= exp2f((max_x_single_warp - max_x_final) * log2_e)
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= exp2f((max_x - max_x_final) * log2_e)
                max_x = max_x_final
            else:
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps
                            * cluster_n
                            * reduction_buffer.element_type.width
                            // 8,
                        )
                if lane_idx < cluster_n:
                    store_shared_remote(
                        f32x2_to_i64(max_x, sum_exp_x),
                        elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                cute.arch.mbarrier_wait(
                    mbar_ptr, phase=phase if phase is not None else 0
                )
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
                max_x_single_warp = cute.make_fragment(num_iter, Float32)
                max_x_single_warp.fill(-Float32.inf)
                sum_exp_x_single_warp = cute.make_fragment(num_iter, Float32)
                sum_exp_x_single_warp.fill(0.0)
                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        max_x_single_warp[i], sum_exp_x_single_warp[i] = i64_to_f32x2(
                            reduction_buffer[row_idx, idx]
                        )
                max_x_final = max_x_single_warp.load().reduce(
                    cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                )
                max_x_final = warp_reduce(max_x_final, cute.arch.fmax)
                sum_exp_x = 0.0
                for i in cutlass.range_constexpr(num_iter):
                    sum_exp_x += sum_exp_x_single_warp[i] * exp2f(
                        (max_x_single_warp[i] - max_x_final) * log2_e
                    )
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= exp2f((max_x - max_x_final) * log2_e)
                max_x = max_x_final
    return max_x, sum_exp_x, (exp_x if cutlass.const_expr(return_exp_x) else None)


@dsl_user_op
def fmin(
    a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range(cute.size(x.shape), unroll_full=True):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def rsqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def silu(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(a) = a * sigmoid(a) = a * (1 + tanh(a / 2)) / 2 = (0.5 * a) * tanh(0.5 * a) + (0.5 * a)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * a, MUFU.TANH, and FFMA.
    """
    a_half = 0.5 * a
    return a_half * tanh(a_half) + a_half


@dsl_user_op
def prmt(a: int | Int32, b: int | Int32, c: int | Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    assert t.element_type.width == 16
    assert (
        cute.size(t.shape) % 4 == 0
    ), "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(
            upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp
        )
        lower0 = cute.arch.shuffle_sync(
            lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp
        )
        t_u32[i * 2 + 0] = prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = prmt(upper0, lower0, selector_lower)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@cute.jit
def fill_oob(
    tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric
) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), 0, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if cutlass.const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(
        vector.extract(
            vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec_i64x1 = vector.from_elements(
        T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip
    )
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    res1 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip
        )
    )
    return res0, res1


@dsl_user_op
def domain_offset_i64(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(
        flat_stride
    ), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(
    idx: cute.typing.Int, tensor: cute.Tensor, dim: int, *, loc=None, ip=None
) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@cute.jit
def warp_prefix_sum(
    val: cutlass.Int32, lane: Optional[cutlass.Int32] = None
) -> cutlass.Int32:
    if cutlass.const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (
                acc_layout_col_major.stride[0][1],
                acc_layout_col_major.stride[1],
            ),  # MMA_M
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@dsl_user_op
def sm90_get_smem_load_op(
    layout_c: cutlass.utils.LayoutEnum,
    elem_ty_c: Type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem load atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_c : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_c : Type[Numeric]
        The element type for output tensor D.

    Returns:
    --------
    Either SmemLoadMatrix or SimtSyncCopy, based on the input parameters.
    """

    if not isinstance(elem_ty_c, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"elem_ty_c must be a Numeric, but got {elem_ty_c}")
    is_m_major = layout_c.is_m_major_c()
    if elem_ty_c.width == 16:
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip
        )
