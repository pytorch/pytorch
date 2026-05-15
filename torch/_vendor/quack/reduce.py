# Copyright (c) 2025, Tri Dao.

import math
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr

from . import utils


@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
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
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
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
        utils.store_shared_remote(
            val,
            utils.elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
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
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
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
    phase: Optional[Int32] = None,
    return_exp_x: bool = False,
) -> [Float32, Float32, Optional[cute.TensorSSA]]:
    assert x.dtype == Float32, "x must be of type Float32"
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n), 2)"""
    max_x = cute.arch.warp_reduction(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
    sum_exp_x = cute.arch.warp_reduction(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            assert reduction_buffer.element_type == Int64, (
                "reduction_buffer must be of type cute.Int64"
            )
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
            if const_expr(mbar_ptr is None):
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx] = utils.f32x2_to_i64(max_x, sum_exp_x)
                cute.arch.barrier()
                max_x_single_warp = -Float32.inf
                sum_exp_x = 0.0
                if lane_idx < warps_per_row:
                    max_x_single_warp, sum_exp_x = utils.i64_to_f32x2(
                        reduction_buffer[row_idx, lane_idx]
                    )
                max_x_final = cute.arch.warp_reduction(max_x_single_warp, cute.arch.fmax)
                sum_exp_x *= cute.math.exp(max_x_single_warp - max_x_final, fastmath=True)
                sum_exp_x = cute.arch.warp_reduction(sum_exp_x, operator.add)
                if const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
            else:
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps * cluster_n * reduction_buffer.element_type.width // 8,
                        )
                if lane_idx < cluster_n:
                    utils.store_shared_remote(
                        utils.f32x2_to_i64(max_x, sum_exp_x),
                        utils.elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
                max_x_single_warp = cute.make_rmem_tensor(num_iter, Float32)
                max_x_single_warp.fill(-Float32.inf)
                sum_exp_x_single_warp = cute.make_rmem_tensor(num_iter, Float32)
                sum_exp_x_single_warp.fill(0.0)
                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        max_x_single_warp[i], sum_exp_x_single_warp[i] = utils.i64_to_f32x2(
                            reduction_buffer[row_idx, idx]
                        )
                max_x_final = max_x_single_warp.load().reduce(
                    cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                )
                max_x_final = cute.arch.warp_reduction(max_x_final, cute.arch.fmax)
                sum_exp_x = 0.0
                for i in cutlass.range_constexpr(num_iter):
                    sum_exp_x += sum_exp_x_single_warp[i] * cute.math.exp(
                        max_x_single_warp[i] - max_x_final, fastmath=True
                    )
                sum_exp_x = cute.arch.warp_reduction(sum_exp_x, operator.add)
                if const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
    return max_x, sum_exp_x, (exp_x if const_expr(return_exp_x) else None)


@cute.jit
def sum_swap_shuffle(
    X: cute.Tensor, elem_per_lane: int = 1, subwarp_size: int = 1, warp_size: int = 32
) -> cute.Tensor:
    """
    For warp reduction, we use Swap Shuffle
    The normal way to reduction among threads:
    use shuffle to let *** the first half of threads *** have *** whole data *** from the second half of threads.
    After each step of reduction, a half of threads won't work in the following steps.
    That is, as the reduction progresses, the efficiency of shuffle & reduction instructions gradually change from 1/2, 1/4 to 1/32 (the worst case).
    To overcome this shortcoming, for a NxN matrix to be reduced among N threads as a 1XN vectors,
    we use swap & shuffle aiming to let *** each half of threads *** have *** a half of data *** from the other half of threads.
    After reduction, each half of threads should deal with a (N/2)x(N/2) sub-matrix independently in the following step.
    We can recursively do this until the problem size is 1.
    """
    assert (
        subwarp_size >= 1
        and subwarp_size <= 32
        and subwarp_size == 1 << int(math.log2(subwarp_size))
    )
    assert (
        warp_size <= 32
        and warp_size % subwarp_size == 0
        and warp_size == 1 << int(math.log2(warp_size))
    )
    lane_idx = cute.arch.lane_idx() // subwarp_size
    X = cute.logical_divide(X, cute.make_layout(elem_per_lane))  # (elem_per_lane, M)
    numvec = cute.size(X, mode=[1])
    assert numvec <= 32 // subwarp_size
    # If X has more values than warp_size // subwarp_size, we first do a normal warp reduction
    # to sum up values held by lanes further than size(X) away
    for i in cutlass.range(
        int(math.log2(numvec)), int(math.log2(warp_size // subwarp_size)), unroll_full=True
    ):
        for v in cutlass.range(cute.size(X), unroll_full=True):
            shfl_val = cute.arch.shuffle_sync_bfly(X[v], offset=(1 << i) * subwarp_size)
            X[v] = X[v] + shfl_val
    for logm in cutlass.range_constexpr(int(math.log2(cute.size(X, mode=[1]))) - 1, -1, -1):
        m = 1 << logm
        for r in cutlass.range(m, unroll_full=True):
            frg_A = X[None, r]
            frg_B = X[None, r + m]
            #  First half of threads swap fragments from the first half of data to the second
            should_swap = not Boolean(lane_idx & m)
            for v in cutlass.range(cute.size(frg_A), unroll_full=True):
                # Step 1: swap
                lower, upper = frg_A[v], frg_B[v]
                frg_A[v] = upper if should_swap else lower
                frg_B[v] = lower if should_swap else upper
                # Step 2: shuffle
                # each half of threads get a half of data from the other half of threads
                shfl_val = cute.arch.shuffle_sync_bfly(frg_A[v], offset=m * subwarp_size)
                # Step 3: reduction
                frg_A[v] = frg_B[v] + shfl_val
    return X[None, 0]
