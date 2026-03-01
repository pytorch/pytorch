"""
Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
CuTE DSL utilities for RMSNorm/LayerNorm kernels from Quack.
"""
# pyre-ignore-all-errors
# pyrefly: ignore-errors
# ruff: noqa: S101

from __future__ import annotations

import operator
from collections.abc import Callable  # noqa: TC003
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op, T


def make_fake_tensor(
    dtype, shape, divisibility=1, leading_dim=-1
) -> Optional[cute.Tensor]:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


@dsl_user_op
def get_copy_atom(
    dtype: type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_2d(
    dtype: type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
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
    val: float | Float32 | Int32 | cutlass.Int64,
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
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), (
        "val must be Float32, Int32, or Int64"
    )
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    asm = (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes"
        f".{suffix} [$0], $1, [$2];"
    )
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            val.ir_value(loc=loc, ip=ip),
            remote_mbar_ptr_i32,
        ],
        asm,
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def fill_oob(
    tXsX: cute.Tensor,
    tXpX: Optional[cute.Tensor],
    fill_value: cute.Numeric,
) -> None:
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@cute.jit
def block_reduce(
    val: cute.Numeric,
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
    return cute.arch.warp_reduction(block_reduce_val, operator.add)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = (
        warp_idx // warps_per_row,
        warp_idx % warps_per_row,
    )
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
            block_reduce_val = operator.add(
                block_reduce_val, reduction_buffer[row_idx, idx]
            )
    return cute.arch.warp_reduction(block_reduce_val, operator.add)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    if const_expr(mbar_ptr is None):
        return block_reduce(val, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(
            val,
            reduction_buffer,
            mbar_ptr,
            phase=phase,
            init_val=init_val,
        )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(cute.ReductionOp.ADD, init_val=init_val, reduction_profile=0)
    else:
        val = x
    val = cute.arch.warp_reduction(
        val,
        operator.add,
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
                val,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
            )
    return val


def get_tiled_copy(
    dtype: type[cutlass.Numeric],
    N: int,
    cluster_n: int,
    threads_per_row: int,
    num_threads: int,
    vecsize: int = 1,
):
    assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
    assert num_threads % cute.arch.WARP_SIZE == 0
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row * cluster_n)
    tiler_mn = (
        num_threads // threads_per_row,
        vecsize * num_blocks_N * threads_per_row,
    )
    tc = tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize)
    return tc, tiler_mn, threads_per_row


def get_reduction_buffer_layout(
    stage: int,
    tv_layout: cute.Layout,
    cluster_n: int,
):
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = (
        num_warps
        if cute.rank(tv_layout.shape[0]) == 1
        else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    )
    return cute.make_ordered_layout(
        (
            num_warps // warps_per_row,
            (warps_per_row, cluster_n),
            stage,
        ),
        order=(1, 0, 2),
    )


def allocate_reduction_buffer_and_mbar(
    smem: cutlass.utils.SmemAllocator,
    reduction_dtype: type[cutlass.Numeric],
    stage: int,
    cluster_n: int,
    tv_layout: cute.Layout,
    is_persistent: bool = False,
) -> tuple[cute.Tensor, Optional[cute.Pointer]]:
    reduction_buffer = smem.allocate_tensor(
        reduction_dtype,
        get_reduction_buffer_layout(stage, tv_layout, cluster_n),
        byte_alignment=8,
    )
    if const_expr(cluster_n > 1):
        mbar_ptr = smem.allocate_array(
            Int64,
            num_elems=(stage if not is_persistent else stage * 2),
        )
    else:
        mbar_ptr = None
    return reduction_buffer, mbar_ptr


@cute.jit
def initialize_cluster(
    tidx: Int32,
    mbar_ptr: cute.Pointer,
    num_warps: int,
    cluster_n: int,
    stage: int,
    is_persistent: bool = False,
):
    if const_expr(cluster_n > 1):
        if tidx < stage:
            cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            if const_expr(is_persistent):
                cute.arch.mbarrier_init(
                    mbar_ptr + stage + tidx,
                    num_warps * cluster_n,
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
