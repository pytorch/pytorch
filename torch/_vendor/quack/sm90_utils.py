# Copyright (c) 2025, Tri Dao.

from typing import Type, Union, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged


# For compatibility with blackwell_helpers.py
make_smem_layout_epi = make_smem_layout


@dsl_user_op
def partition_for_epilogue(
    cT: cute.Tensor,
    epi_tile: cute.Tile,
    tiled_copy: cute.TiledCopy,
    tidx: Int32,
    reference_src: bool,  # do register tensors reference the src or dst layout of the tiled copy
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    thr_copy = tiled_copy.get_slice(tidx)
    cT_epi = cute.flat_divide(cT, epi_tile)
    # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
    if const_expr(reference_src):
        return thr_copy.partition_S(cT_epi, loc=loc, ip=ip)
    else:
        return thr_copy.partition_D(cT_epi, loc=loc, ip=ip)


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    # A_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if const_expr(swap_AB):
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        # We make a new mma_atom since we'll be modifying its attribute (accumulate).
        # Otherwise the compiler complains "operand #0 does not dominate this use"
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    if const_expr(swap_AB):
        return gemm_zero_init(
            tiled_mma, shape[::-1], tCrB, tCrA, B_idx, A_idx, wg_wait, swap_AB=False
        )
    else:
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
        return acc


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    if const_expr(swap_AB):
        gemm_w_idx(tiled_mma, acc, tCrB, tCrA, zero_init, B_idx, A_idx, wg_wait, swap_AB=False)
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool = False,
):
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        if const_expr(not is_rs):
            assert sA is not None
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        else:
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2])))
        assert sB is not None
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        if const_expr(not is_rs):
            assert sB is not None
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_A(sB))
        else:  # B in rmem
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[1], shape_mnk[2])))
        assert sA is not None
        tCrA = thr_mma.make_fragment_B(thr_mma.partition_B(sA))
    return acc, tCrA, tCrB
