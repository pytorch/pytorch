# Copyright (c) 2025, Tri Dao.

from typing import Literal, Type, Union, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from torch._vendor.quack import copy_utils


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


def choose_sm90_wgmma_layout_mn(
    tile_m: int,
    tile_n: int,
    num_wg: int,
    *,
    allow_swap_ab: bool = True,
) -> tuple[bool, int]:
    """Return ``(swap_AB, AtomLayoutM)`` minimizing Hopper SS WGMMA SMEM traffic.

    The logical MMA is ``(tile_m, tile_n)``.  Hopper's physical WGMMA M mode is
    64, and for ``num_wg`` in {1, 2, 3} the only useful warp-group layouts are
    all WGs along physical M or all WGs along physical N.  The returned
    ``AtomLayoutM`` is in the caller's logical coordinate system.  Callers that
    pass a physical ``atom_layout_mnk`` directly to a lower-level MMA builder
    should swap the first two atom-layout modes when ``swap_AB`` is true.
    """
    if num_wg not in (1, 2, 3):
        raise ValueError(f"SM90 WGMMA layout chooser expects num_wg in {{1, 2, 3}}, got {num_wg}")
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"tile_m and tile_n must be positive, got {(tile_m, tile_n)}")

    def best_physical_layout(x: int, y: int) -> tuple[int, int] | None:
        # Prefer split-M when valid: it has wg_n=1, strictly lower traffic than
        # split-N for the same orientation when num_wg > 1.
        if x % (64 * num_wg) == 0 and y % 8 == 0:
            return (num_wg, 1)
        if x % 64 == 0 and y % (8 * num_wg) == 0:
            return (1, num_wg)
        return None

    best: tuple[int, bool, int] | None = None
    for swap_ab in (False, True) if allow_swap_ab else (False,):
        physical_m, physical_n = (tile_n, tile_m) if swap_ab else (tile_m, tile_n)
        layout = best_physical_layout(physical_m, physical_n)
        if layout is None:
            continue
        physical_atom_m, physical_atom_n = layout
        score = physical_m * physical_atom_n
        atom_layout_m = physical_atom_n if swap_ab else physical_atom_m
        candidate = (score, swap_ab, atom_layout_m)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(
            "no valid SM90 WGMMA layout for "
            f"tile_m={tile_m}, tile_n={tile_n}, num_wg={num_wg}, "
            f"allow_swap_ab={allow_swap_ab}"
        )
    _, swap_ab, atom_layout_m = best
    return swap_ab, atom_layout_m


def make_tiled_mma(
    a_dtype: Type[Numeric],
    a_major: Literal["K", "MN"],
    b_major: Literal["K", "MN"],
    tiler_n: int,
    source: Literal["SS", "RS"] = "SS",
    atom_layout_mnk: tuple = (1, 1, 1),
    swap_AB: bool = False,
    b_dtype: Optional[Type[Numeric]] = None,
    acc_dtype: Type[Numeric] = Float32,
) -> cute.TiledMma:
    """`b_dtype` defaults to `a_dtype`; pass it for mixed-precision MMAs (e.g. fp8).
    `acc_dtype` defaults to Float32."""
    if b_dtype is None:
        b_dtype = a_dtype
    mode = {"K": cute.nvgpu.OperandMajorMode.K, "MN": cute.nvgpu.OperandMajorMode.MN}
    a_mode, b_mode = mode[a_major], mode[b_major]
    if swap_AB:
        a_mode, b_mode = b_mode, a_mode
    a_source = warpgroup.OperandSource.RMEM if source == "RS" else warpgroup.OperandSource.SMEM
    return sm90_utils_og.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_mode,
        b_mode,
        acc_dtype,
        atom_layout_mnk=atom_layout_mnk,
        tiler_mn=(64, tiler_n),
        a_source=a_source,
    )


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


def canonical_a_load_s2r(
    tiled_mma, sA, tidx, tCrA, position_independent=False, transpose=None, atom=None
):
    """The canonical A-operand produce for the RS mainloop: an ldmatrix tiled
    copy derived from the MMA (LDSM for k-major A, LDSM.T for m-major — one
    code path, 16-bit only). Returns ``copy_block(stage_idx, b, k_tile)``,
    which s2r loads k16 block b (a static Python int) of pipeline stage
    stage_idx into the fragment (``k_tile``, the global k-tile index the
    mainloop threads through the seam for coordinate-dependent transforms,
    is unused here). This is the produce seam: transforms substitute their own
    copy_block (e.g. LDS + dequant) while the mainloop keeps owning the WGMMA
    issue and commit-group discipline. Also serves the SM120 warp-MMA mainloop
    (fragment atoms are LDSM-identical); its MmaF16BF16Op carries no major mode
    (operand layout fixed K-major, the smem major only picks LDSM vs LDSM.T),
    so ``transpose`` must be passed explicitly there."""
    if transpose is None:
        transpose = tiled_mma.op.a_major_mode == cute.nvgpu.OperandMajorMode.MN
    if const_expr(atom is None):
        atom = copy_utils.get_smem_load_atom(sA.element_type, transpose)
    smem_tiled_copy_A = cute.make_tiled_copy_A(atom, tiled_mma)
    thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    # (CPY, CPY_M, CPY_K, STAGE); position-independent partition absorbs the
    # swizzle into the pointer, so per-block addresses are linear (plain IMAD
    # chains ptxas can hoist) instead of a SHF+LOP3 XOR per LDSM
    if const_expr(position_independent):
        tCsA_copy_view = copy_utils.partition_S_position_independent(thr_copy_A, sA)
    else:
        tCsA_copy_view = thr_copy_A.partition_S(sA)
    tCrA_copy_view = thr_copy_A.retile(tCrA)

    def copy_block(stage_idx, b, k_tile=None):
        cute.copy(
            smem_tiled_copy_A,
            tCsA_copy_view[None, None, b, stage_idx],
            tCrA_copy_view[None, None, b],
        )

    return copy_block
