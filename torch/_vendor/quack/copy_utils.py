# Copyright (c) 2025-2026, QuACK team.

from typing import Optional, Type, Tuple, Callable, Sequence
from functools import partial

import cutlass
import cutlass.cute as cute

from cutlass import Int32, Int16, Boolean, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, tcgen05, warp, warpgroup
from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup  # noqa
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

from . import layout_utils
from .utils import make_vector


Sm100MmaPeerBitMask = 0xFEFFFFFF


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def sr_cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    seed: Int32,
    tidx: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Like cvt_copy but uses stochastic rounding for FP32 -> BF16 conversion."""
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    from .rounding import convert_f32_to_bf16_sr
    from cutlass.cute.tensor import TensorSSA

    src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
    src_vec = src.load()
    raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tidx, loc=loc, ip=ip)
    src_cvt.store(TensorSSA(raw_vec, src_vec.shape, dst.element_type))
    src = src_cvt
    cute.copy(tiled_copy, src, dst, loc=loc, ip=ip)


@dsl_user_op
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_rmem_tensor_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def contiguous(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_rmem_tensor(src.shape, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def load_s2r_retile(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst_shape: cute.Tensor | cute.Shape,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    # Will also accept dst_shape being a tensor, in which case we write into that tensor
    if const_expr(not isinstance(dst_shape, cute.Tensor)):
        dst = cute.make_rmem_tensor(dst_shape, src.element_type, loc=loc, ip=ip)
    else:
        dst = dst_shape
    cute.copy(tiled_copy, src, tiled_copy.retile(dst), loc=loc, ip=ip)
    return dst


@dsl_user_op
def load_t2r(
    thr_copy: cute.ThrCopy, shape: cute.Shape, src: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    cDst = cute.make_identity_tensor(shape)
    dst = cute.make_rmem_tensor(thr_copy.partition_D(cDst).shape, src.element_type, loc=loc, ip=ip)
    cute.copy(thr_copy, src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
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


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
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
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


# def tiled_copy_2d(
#     dtype: Type[cutlass.Numeric], major_mode_size: int, num_threads: int, is_async: bool = False
# ) -> cute.TiledCopy:
#     num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
#     copy_elems = num_copy_bits // dtype.width
#     copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
#     copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
#     gmem_threads_per_row = major_mode_size // copy_elems
#     assert num_threads % gmem_threads_per_row == 0
#     thr_layout = cute.make_ordered_layout(
#         (num_threads // gmem_threads_per_row, gmem_threads_per_row),
#         order=(1, 0),
#     )
#     val_layout = cute.make_layout((1, copy_elems))
#     return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


# Ragged tensor trick for TMA: encodes variable-length sequences into a higher-rank
# tensor so that TMA's out-of-bounds checking handles sequence boundaries.
#
# Given a tensor T with a ragged dimension (variable-length across batches), we create
# a higher-rank tensor where the ragged dim is replaced with a fixed size `big_int`, and
# extra dim(s) are appended. When indexing into a specific sequence at (offset, length),
# `offset_ragged_tensor` computes coordinates such that:
#   ragged_coord = big_int - length   (OOB check clamps reads past the sequence end)
#   extra_coord(s) = f(offset, length) (selects the correct memory region)
#
# ptr_shift=True: 1-extra-dim approach (adds 1 dim, supports up to 4D input):
#   Shape:  (*before, big_int, *after, max_int)
#   Stride: (*original_strides, stride_r)     where stride_r = T.stride[ragged_dim]
#   Pointer shifted backward by big_int * stride_r elements.
#   Address for coords (big_int - length) in ragged dim, (offset + length) in extra dim:
#     addr = (base - big_int * s_r) + (big_int - length) * s_r + (offset + length) * s_r
#          = base + offset * s_r                                                      [correct]
#   Works for epilogue TMA store. Does NOT work for TMA load with large big_int
#   — the shifted pointer must land in physically mapped GPU memory.
#
# ptr_shift=False: 2-extra-dim approach (adds 2 dims, supports up to 3D input):
#   Shape:  (*before, big_int, *after, max_int, max_int)
#   Stride: (*before_strides, stride_r, *after_strides, 2^34 - stride_r, stride_r)
#   No pointer shift. Uses 64-bit address wraparound to cancel the ragged offset.
#   Let W = 2^34 - stride_r. Address for coords (big_int - length) in ragged dim,
#   big_int in extra dim 0, (offset + length) in extra dim 1:
#     addr = base + (big_int - length) * s_r + big_int * W + (offset + length) * s_r
#          = base + big_int * (s_r + W) - length * s_r + (offset + length) * s_r
#          = base + big_int * 2^34 + offset * s_r
#   Since big_int = 2^30: big_int * 2^34 = 2^64 ≡ 0 (mod 2^64), so:
#     addr = base + offset * s_r                                                      [correct]
#   Works for all TMA paths since the base pointer is never shifted.
#
# Ragged tensor was adapted from the implementation from Triton, but here we have an option that
# only needs 1 extra dimension instead of 2.
# https://github.com/triton-lang/triton/blob/main/python/triton/tools/ragged_tma.py
BIG_INT = 2**30
MAX_INT = 2**31 - 1
BIG_INT_INV = 2**64 // BIG_INT


@dsl_user_op
def create_ragged_tensor_for_tma(
    T: cute.Tensor,
    ragged_dim: int = 0,
    ptr_shift: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    rank = cute.rank(T)
    if ragged_dim < 0:
        ragged_dim += rank
    if ptr_shift:
        assert rank <= 4, "ptr_shift ragged tensor only supports up to 4 dimensions"
        new_shape = T.shape[:ragged_dim] + (BIG_INT,) + T.shape[ragged_dim + 1 :] + (MAX_INT,)
        new_stride = T.stride + (T.stride[ragged_dim],)
        ptr_offset = (None,) * ragged_dim + (-BIG_INT,) + (None,) * (rank - ragged_dim - 1)
        new_ptr = cute.domain_offset(ptr_offset, T).iterator
        return cute.make_tensor(new_ptr, cute.make_layout(new_shape, stride=new_stride))
    else:
        assert rank <= 3, "non-ptr_shift ragged tensor only supports up to 3 dimensions"
        stride_r = T.stride[ragged_dim]
        new_shape = (
            T.shape[:ragged_dim] + (BIG_INT,) + T.shape[ragged_dim + 1 :] + (MAX_INT, MAX_INT)
        )
        new_stride = (
            T.stride[:ragged_dim]
            + (stride_r,)
            + T.stride[ragged_dim + 1 :]
            + (BIG_INT_INV - stride_r, stride_r)
        )
        return cute.make_tensor(T.iterator, cute.make_layout(new_shape, stride=new_stride))


@dsl_user_op
def offset_ragged_tensor(
    T: cute.Tensor,
    offset: Int32,
    length: Int32,
    ragged_dim: int = 0,
    ptr_shift: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    rank = cute.rank(T)
    if ragged_dim < 0:
        ragged_dim += rank
    big_int = cute.size(T, mode=[ragged_dim])
    offset_val = big_int - length
    if ptr_shift:
        # 1-extra-dim: rank = original_rank + 1
        assert rank >= ragged_dim + 2
        offset_tuple = (None,) * ragged_dim + (offset_val,) + (None,) * (rank - ragged_dim - 2)
        index_tuple = (None,) * (rank - 1) + (offset + length,)
    else:
        # 2-extra-dim: rank = original_rank + 2, last 2 modes are the wraparound dims
        assert rank >= ragged_dim + 3
        offset_tuple = (None,) * ragged_dim + (offset_val,) + (None,) * (rank - ragged_dim - 3)
        index_tuple = (None,) * (rank - 2) + (big_int, offset + length)
    return cute.domain_offset(offset_tuple, T[index_tuple])


def swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)


def swizzle_ptr(ptr: cute.Pointer):
    swz = ptr.type.swizzle_type
    ptr_int = swizzle_int(ptr.toint(), swz.num_bits, swz.num_base, swz.num_shift)
    return cute.make_ptr(ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)


def as_position_independent_swizzle_tensor(tensor: cute.Tensor) -> cute.Tensor:
    outer = tensor.layout
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    inner = cute.make_swizzle(swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift)
    # Need to recast the swizzle from byte (e.g. <3, 4, 3> to element units (e.g. <3, 3, 3> for
    # for 16 bits and <3, 2, 3> for 32 bits)
    new_layout = cute.recast_layout(
        width, 8, cute.make_composed_layout(inner, 0, cute.recast_layout(8, width, outer))
    )
    # recast_ptr to remove the pointer swizzle
    return cute.make_tensor(cute.recast_ptr(tensor.iterator, dtype=tensor.element_type), new_layout)


def partition_D_position_independent(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(as_position_independent_swizzle_tensor(tensor)).layout,
    )


def partition_S_position_independent(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_S(tensor).iterator),
        thr_copy.partition_S(as_position_independent_swizzle_tensor(tensor)).layout,
    )


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
        return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip)
    else:
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip)


def get_smem_store_atom(
    element_type: Type[cute.Numeric],
    transpose: bool = False,
    major_mode_size: Optional[int] = None,
) -> cute.CopyAtom:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if const_expr(arch < Arch.sm_90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
        )
    else:
        num_matrices = (
            4
            if major_mode_size is None or major_mode_size % 16 == 0
            else (2 if major_mode_size % 8 == 0 else 1)
        )
        return cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
            element_type,
        )


def get_smem_load_atom(
    element_type: Type[cute.Numeric],
    transpose: bool = False,
    major_mode_size: Optional[int] = None,
) -> cute.CopyAtom:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if const_expr(arch < Arch.sm_90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
        )
    else:
        num_matrices = (
            4
            if major_mode_size is None or major_mode_size % 16 == 0
            else (2 if major_mode_size % 8 == 0 else 1)
        )
        return cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
            element_type,
        )


def get_smem_store_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    transpose: bool = False,
    position_independent=False,
    major_mode_size: Optional[int] = None,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = get_smem_store_atom(dtype, transpose, major_mode_size=major_mode_size)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tRS_sC = thr_copy.partition_D(sC)
    else:
        tRS_sC = partition_D_position_independent(thr_copy, sC)

    def copy_fn(src: cute.Tensor, dst_idx: Optional[Int32] = None, **new_kwargs):
        dst_tensor = tRS_sC if const_expr(dst_idx is None) else tRS_sC[None, None, None, dst_idx]
        cvt_copy(tiled_copy, src, dst_tensor, retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sC


def get_smem_load_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    transpose: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = get_smem_load_atom(dtype, transpose)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tSR_sC = thr_copy.partition_S(sC)
    else:
        tSR_sC = partition_S_position_independent(thr_copy, sC)
    copy_atom_RS = get_smem_store_atom(dtype, transpose)
    thr_copy_RS = cute.make_tiled_copy_C(copy_atom_RS, tiled_mma).get_slice(tidx)
    tRS_shape = thr_copy_RS.partition_S(cute.make_identity_tensor(sC.shape[:2])).shape

    def copy_fn(src_idx: Optional[Int32] = None, **new_kwargs):
        src_tensor = tSR_sC if const_expr(src_idx is None) else tSR_sC[None, None, None, src_idx]
        return load_s2r_retile(tiled_copy, src_tensor, dst_shape=tRS_shape, **new_kwargs)

    return copy_fn, thr_copy, tSR_sC


def epilog_smem_copy_atom(
    tiled_mma: cute.TiledMma, epi_tile: cute.Shape, transpose: bool = False
) -> cute.TiledCopy:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if const_expr(arch < Arch.sm_90):
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float16,  # this is just to get the right source layout
            num_bits_per_copy=(2 if not transpose else 1) * cutlass.Float16.width,
        )
    else:
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose, num_matrices=4 if epi_tile[1] % 16 == 0 else 2),
            cutlass.Float16,  # this is just to get the right source layout
        )
    tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
    return tiled_copy_C_atom


def get_smem_store_epi(
    tiled_mma: cute.TiledMma,
    epi_tile: cute.Shape,
    sC: Optional[cute.Tensor],
    tidx: Int32,
    transpose: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor, cute.Tensor]:
    dtype = sC.element_type if const_expr(sC is not None) else cutlass.Float16
    copy_atom = get_smem_store_atom(dtype, transpose)
    tiled_copy_C_atom = epilog_smem_copy_atom(tiled_mma, epi_tile)
    tiled_copy = cute.make_tiled_copy_S(copy_atom, tiled_copy_C_atom)
    thr_copy = tiled_copy.get_slice(tidx)
    tRS_sC = None
    if const_expr(sC is not None):
        if const_expr(not position_independent):
            tRS_sC = thr_copy.partition_D(sC)
        else:
            tRS_sC = partition_D_position_independent(thr_copy, sC)
    sC_shape = sC.shape[:2] if sC is not None else epi_tile
    # (R2S, R2S_M, R2S_N, PIPE_C)
    tRS_rC_shape = thr_copy.partition_S(cute.make_identity_tensor(sC_shape)).shape
    tRS_rC = cute.make_rmem_tensor(tRS_rC_shape, tiled_mma.op.acc_dtype)

    def copy_fn(src: cute.Tensor, dst_idx: Int32, **new_kwargs):
        cvt_copy(tiled_copy, src, tRS_sC[None, None, None, dst_idx], **new_kwargs)

    return copy_fn if const_expr(sC is not None) else None, thr_copy, tRS_sC, tRS_rC


def get_smem_store_A(
    tiled_mma: cute.TiledMma, sA: cute.Tensor, tidx: Int32, position_independent=False
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sA.element_type
    transpose = tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
    copy_atom = get_smem_store_atom(dtype, transpose)
    tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tRS_sA = thr_copy.partition_D(sA)
    else:
        tRS_sA = partition_D_position_independent(thr_copy, sA)

    def copy_fn(src: cute.Tensor, dst_idx: Int32, **new_kwargs):
        cvt_copy(tiled_copy, src, tRS_sA[None, None, None, dst_idx], retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sA


def get_smem_load_A(
    tiled_mma: cute.TiledMma,
    sA: cute.Tensor,
    tidx: Int32,
    with_dst_tensor: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sA.element_type
    transpose = tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
    copy_atom = get_smem_load_atom(dtype, transpose)
    tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tSR_sA = thr_copy.partition_S(sA)
    else:
        tSR_sA = partition_S_position_independent(thr_copy, sA)
    tRS_shape = tiled_mma.partition_shape_A(sA.shape[:2])

    def copy_fn(src_idx: Int32, **new_kwargs):
        return load_s2r_retile(
            tiled_copy, tSR_sA[None, None, None, src_idx], dst_shape=tRS_shape, **new_kwargs
        )

    def copy_fn_w_dst_tensor(src_idx: Int32, dst: cute.Tensor, **new_kwargs):
        return load_s2r_retile(tiled_copy, tSR_sA[None, None, None, src_idx], dst, **new_kwargs)

    return copy_fn if not with_dst_tensor else copy_fn_w_dst_tensor, thr_copy, tSR_sA


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    # cache_hint = cutlass.Int64(0x14F0000000000000)  # EVICT_LAST
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        # [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value(), cache_hint.ir_value()],
        # "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [$0], [$1], $2, $3;",
        # "l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def get_tma_desc_addr(tma_atom: cute.CopyAtom, *, loc=None, ip=None) -> cute.Pointer:
    """
    Get the address of the TMA descriptor embedded in a TMA Copy Atom.

    Extracts the constant memory address of the TMA descriptor for use with
    custom PTX instructions.

    :param tma_atom: TMA Copy Atom from make_tiled_tma_atom
    :return: Pointer to TMA descriptor in constant memory

    Example:
        >>> desc_ptr = get_tma_descriptor_address(tma_atom)
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    tma_desc_ptr_type = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
    )
    return _cute_nvgpu_ir.get_tma_desc_addr(tma_desc_ptr_type, exec_atom, loc=loc, ip=ip)


@dsl_user_op
def tma_gather4_load(
    tma_desc_ptr: cute.Pointer,
    dst_smem_ptr: cute.Pointer,
    mbarrier_ptr: cute.Pointer,
    col_idx: Int32,
    row_indices: Sequence[Int32],
    *,
    num_cta: int = 1,
    multicast_mask=None,
    loc=None,
    ip=None,
) -> None:
    """
    Perform TMA gather4 load from global memory to shared memory.

    Issues PTX instruction:
    cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes
        [dstMem], [tensorMap, {col_idx, row0, row1, row2, row3}], [smem_bar];

    This loads 4 rows (specified by row_indices) from a 2D tensor at the given
    column index into shared memory, using the TMA descriptor.

    :param tma_desc_ptr: Pointer to TMA descriptor in constant memory (128-byte aligned)
    :type tma_desc_ptr:  Pointer
    :param dst_smem_ptr: Destination address in shared memory
    :type dst_smem_ptr:  Pointer
    :param mbarrier_ptr: Pointer to mbarrier in shared memory for completion tracking
    :type mbarrier_ptr:  Pointer
    :param col_idx:      Column index
    :type col_idx:       Int32
    :param row_indices:  Sequence of exactly 4 row indices
    :type row_indices:   Sequence[Int32]
    :param num_cta:      Number of CTAs participating (default: 1)
    :type num_cta:       int
    :param multicast_mask: Optional multicast mask
    :type multicast_mask: Int16

    Requirements:
        - row_indices must contain exactly 4 elements
        - Compute capability >= SM_100 (Blackwell)
        - TMA descriptor must be properly initialized for 2D tensor

    Example:
        >>> from cutlass.cute.nvgpu import cpasync
        >>> from cutlass.cute import core
        >>>
        >>> # Create TMA descriptor
        >>> tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(...)
        >>> tma_desc_ptr = get_tma_descriptor_address(tma_atom)
        >>>
        >>> # Compute indices (typically from kernel logic)
        >>> col_idx = core.get(...) or 5  # Int32 value
        >>> row_indices = [core.get(...) for _ in range(4)]  # 4 Int32 values
        >>>
        >>> # Gather 4 rows at computed column
        >>> tma_gather4_load(
        ...     tma_desc_ptr=tma_desc_ptr,
        ...     dst_smem_ptr=smem_ptr,
        ...     mbarrier_ptr=barrier_ptr,
        ...     col_idx=col_idx,
        ...     row_indices=row_indices
        ... )
    """
    if len(row_indices) != 4:
        raise ValueError(f"gather4 requires exactly 4 row indices, got {len(row_indices)}")
    col_val = Int32(col_idx).ir_value()
    row_vals = [Int32(row_idx).ir_value() for row_idx in row_indices]
    # Convert pointers to integer addresses
    desc_addr = tma_desc_ptr.toint(loc=loc, ip=ip).ir_value()
    dst_addr = dst_smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_addr = mbarrier_ptr.toint(loc=loc, ip=ip)
    if num_cta > 1:
        # Executed by both CTAs. Set peer bit to 0 so that the
        # transaction bytes will update CTA0's barrier.
        mbar_addr = mbar_addr & Sm100MmaPeerBitMask
    mbar_addr = mbar_addr.ir_value()
    # Handle multicast_mask - may already be ir.Value or Python int
    multicast_mask_val = None
    if multicast_mask is not None:
        multicast_mask_val = Int16(multicast_mask).ir_value()
    assert multicast_mask_val is None, "multicast is not supported yet"
    # Emit inline PTX for TMA gather4
    # PTX: cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes
    #      [dstMem], [tensorMap, {col, row0, row1, row2, row3}], [smem_bar];
    ptx = (
        f"cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::{num_cta} "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7];"
    )

    llvm.inline_asm(
        None,
        [
            dst_addr,
            desc_addr,
            col_val,
            row_vals[0],
            row_vals[1],
            row_vals[2],
            row_vals[3],
            mbar_addr,
        ],
        ptx,
        "r,l,r,r,r,r,r,r",  # constraints: register, long, 6x register
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    group_rank_src = const_expr(cute.rank(src_tensor) - (1 if not single_stage else 0))
    group_rank_dst = const_expr(cute.rank(dst_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    src = cute.group_modes(src_tensor, 0, group_rank_src)
    dst = cute.group_modes(dst_tensor, 0, group_rank_dst)

    def copy_bulk(src_idx, dst_idx, tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(
                atom,
                src[None, src_idx],
                dst[None, dst_idx],
                mbar_ptr=tma_bar_ptr,
                **new_kwargs,
                **kwargs,
            )

    def copy_bulk_single_stage(tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(atom, src, dst, mbar_ptr=tma_bar_ptr, **new_kwargs, **kwargs)

    return copy_bulk if const_expr(not single_stage) else copy_bulk_single_stage


@dsl_user_op
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
        loc=loc,
        ip=ip,
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(
            atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs, loc=loc, ip=ip
        )

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def s2t_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    cta_group: tcgen05.CtaGroup,
) -> Callable:
    """
    Make tiledCopy for smem to tmem load, then return a copy function over stages.

    :param src_tensor: The source tensor in smem
    :param dst_tensor: The destination tensor in tmem
    """
    assert src_tensor.element_type == dst_tensor.element_type
    # (MMA, MMA_MN, MMA_K, STAGE)
    src_compact = cute.filter_zeros(src_tensor)
    # (MMA, MMA_MN, MMA_K)
    dst_compact = cute.filter_zeros(dst_tensor)
    # Make S2T CopyAtom and tiledCopy.
    copy_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), dst_tensor.element_type)
    tiled_copy = tcgen05.make_s2t_copy(copy_atom, dst_compact)
    thr_copy = tiled_copy.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    src_partition = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, thr_copy.partition_S(src_compact))
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    dst_partition = thr_copy.partition_D(dst_compact)

    @dsl_user_op
    def copy_s2t(stage_idx, *, loc=None, ip=None, **new_kwargs):
        # Stage slice of partitioned source tensor: ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        stage_coord = (None, None, None, None, stage_idx)
        cute.copy(
            tiled_copy, src_partition[stage_coord], dst_partition, loc=loc, ip=ip, **new_kwargs
        )

    return copy_s2t


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn


@cute.jit
def gather_m_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (whatever, K)
    sA: cute.Tensor,  # (tile_M, tile_K, STAGE)
    gsAIdx: cute.Tensor,  # (tile_M), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    tile_M, tile_K = cute.size(sA, mode=[0]), cute.size(sA, mode=[1])
    tAsA = partition_D_position_independent(thr_copy_A, sA)
    # k-major
    assert tAsA.shape[2] == 1
    tAsA = cute.group_modes(cute.slice_(tAsA, (None, None, 0, None)), 0, 2)

    is_even_m_smem = tile_M % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_M)
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor((tile_M, tile_K))
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_rmem_tensor(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    m_idx = cute.make_rmem_tensor(rows_per_thread, Int32)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        row_idx = tAcA[0, m, 0][0]
        if tApA_m[m]:
            m_idx[m] = gsAIdx[row_idx]
        else:
            m_idx[m] = 0  # It's ok to load row 0 in the case of OOB

    mA_k = cute.logical_divide(mA, (None, tile_K))

    def copy_fn(src_idx, dst_idx, pred: cutlass.Constexpr[bool] = False):
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_K
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        mA_cur = mA_k[None, (None, src_idx)]
        for m in cutlass.range_constexpr(tAcA.shape[1]):
            # cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,)) would give shape
            # ((elems_per_load), thread_per_row)
            # But we actually want shape ((elems_per_load, 1), thread_per_row) to match tAsA
            # So we append 1s to the last dimension and then do tiled_divide, then slice.
            mA_row = cute.tiled_divide(
                cute.append_ones(mA_cur[m_idx[m], None], up_to_rank=2), (elems_per_load, 1)
            )[None, None, 0]
            if const_expr(is_even_m_smem) or tApA_m[m]:
                # There's only 1 load per row
                assert cute.size(tAcA.shape, mode=[2]) == 1
                ki = tAcA[0, 0, 0][1] // elems_per_load
                cute.copy(thr_copy_A, mA_row[None, ki], tAsA[(None, m), dst_idx], pred=tApA_k)

    return copy_fn


@cute.jit
def gather_k_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (tile_M, whatever)
    sA: cute.Tensor,  # (tile_M, tile_K, STAGE)
    gsAIdx: cute.Tensor,  # (tile_K, RestK), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    gAIdx, sAIdx = None, None
    if const_expr(gsAIdx.memspace == cute.AddressSpace.gmem):
        gAIdx = gsAIdx
    else:
        assert gsAIdx.memspace == cute.AddressSpace.smem
        sAIdx = gsAIdx
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    # (atom_v, CPY_M, 1, STAGE)
    tAsA = thr_copy_A.partition_D(sA)
    # m-major
    tAsA = cute.group_modes(tAsA, 0, 3)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_rmem_tensor(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    threads_per_col = const_expr(thr_copy_A.tiler_mn[0].shape // elems_per_load)
    # This is very convoluted but idk a better way
    # for tile_M=128, flat_divide gives (8, 16, K),
    # then logical_divide gives ((8, 1), (8, 2), K).
    tidx = thr_copy_A.thr_idx
    tAmA = cute.logical_divide(
        cute.flat_divide(mA, (elems_per_load,)), (elems_per_load, threads_per_col)
    )[None, (tidx % threads_per_col, None), None]  # ((8, 1), 2, K)

    def prefetch_from_gmem_fn(src_idx, pred: bool = False) -> Tuple[cute.Tensor, cute.Tensor]:
        # Prefetch mAIdx early, even before smem is free
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        gAIdx_cur = gAIdx[None, src_idx]
        k_idx = cute.make_rmem_tensor(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            if const_expr(not pred):
                k_idx[k] = gAIdx_cur[col_idx]
            else:
                if tApA_k[k]:
                    k_idx[k] = gAIdx_cur[col_idx]
                else:
                    k_idx[k] = -1
        return k_idx, tApA_k

    def prefetch_from_smem_fn(
        a_prefetch_pipeline, src_idx, dst_idx, a_prefetch_consumer_state, pred: bool = False
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
        sAIdx_cur = sAIdx[None, dst_idx]
        k_idx = cute.make_rmem_tensor(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            k_idx[k] = sAIdx_cur[col_idx]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
        return k_idx, tApA_k

    def copy_fn(
        src_idx, dst_idx, k_idx_tApA_k: Tuple[cute.Tensor, cute.Tensor], pred: bool = False
    ):
        k_idx, tApA_k = k_idx_tApA_k
        tApA_k_pred = None
        if const_expr(pred):
            tApA_k_pred = cute.prepend_ones(tApA_k, up_to_rank=2)  # (1, cols_per_thread)
        for k in cutlass.range_constexpr(tAcA.shape[2]):
            # copy_A(tAmA[None, None, k_idx[k]], tAsA[(None, None, k), smem_idx], pred=cute.prepend_ones(tApA_m, up_to_rank=2))
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                if tApA_m[m]:
                    cute.copy(
                        thr_copy_A,
                        tAmA[None, m, k_idx[k]],
                        tAsA[(None, m, k), dst_idx],
                        pred=None if const_expr(tApA_k_pred is None) else tApA_k_pred[None, k],
                    )

    return copy_fn, prefetch_from_gmem_fn if const_expr(
        gAIdx is not None
    ) else prefetch_from_smem_fn


@cute.jit
def gather_m_get_tma_copy_fn(
    tma_atom: cute.CopyAtom,
    mA: cute.Tensor,  # (whatever, K)
    sA: cute.Tensor,  # ((4, 32), (64, 1), STAGE)
    sAIdx: cute.Tensor,  # (tile_M),
    warp_idx: Int32,
    num_warps: int,
    num_cta: int = 1,
) -> Callable:
    tile_M = cute.size(sAIdx, mode=[0])
    tile_K = cute.size(sA[None, None, 0]) // tile_M
    assert tile_M % 4 == 0
    # cta_group = 1 if tma_atom.op.cta_group == CtaGroup.ONE else 2
    cta_group = num_cta  # Somehow all tma_atom has CtaGroup.ONE inside the kernel

    copy_AIdx_s2r = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128),
        cute.make_layout(num_warps),  # thr_layout
        cute.make_layout(4),  # val_layout
    )
    warp_copy_AIdx_s2r = copy_AIdx_s2r.get_slice(warp_idx)
    tSR_sAIdx = warp_copy_AIdx_s2r.partition_S(sAIdx)
    # ((4, 1), 8, (64, 1), STAGE)
    tSR_sA = warp_copy_AIdx_s2r.partition_S(sA)
    tSR_rAIdx = load_s2r(tSR_sAIdx)
    tma_desc_ptr = get_tma_desc_addr(tma_atom)
    tma_gather4_load_fn = partial(tma_gather4_load, tma_desc_ptr, num_cta=cta_group)

    def copy_fn(src_idx, dst_idx, tma_bar_ptr: cute.Pointer):
        tSR_sA_cur = tSR_sA[None, None, None, dst_idx]
        col_idx = tile_K * src_idx
        for m in cutlass.range(cute.size(tSR_rAIdx, mode=[1]), unroll_full=True):
            row_indices = [tSR_rAIdx[v, m] for v in range(4)]
            smem_ptr = tSR_sA_cur[None, m, None].iterator
            with cute.arch.elect_one():
                tma_gather4_load_fn(smem_ptr, tma_bar_ptr, col_idx, row_indices)

    return copy_fn


@cute.jit
def gather_k_get_tma_copy_fn(
    tma_atom: cute.CopyAtom,
    sA: cute.Tensor,  # ((4, tile_K/4), (tile_M,), STAGE) — K-grouped load layout
    sAIdx: cute.Tensor,  # (tile_K, a_prefetch_stage) — K indices in smem
    col_idx: Int32,  # M offset in global tensor (contiguous dim for M-major)
    warp_idx: Int32,
    num_warps: int,
    num_cta: int = 1,
) -> Tuple[Callable, Callable]:
    """Build a copy function for TMA gather4 in K dimension (M-major A).

    Each gather4 instruction loads 4 K-columns × tile_M contiguous M-elements.
    col_idx is the absolute M position in the global tensor.
    K indices come from sAIdx (prefetched to smem by the scheduler warp).

    Returns copy_fn(src_idx, dst_idx, tma_bar_ptr) which:
      Issues gather4 calls with those K indices as row_indices
    """
    tile_K = cute.size(sAIdx, mode=[0])
    assert tile_K % 4 == 0
    cta_group = num_cta

    # Tiled copy for loading K indices from smem to registers (4 per vector, across warps)
    copy_AIdx_s2r = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128),
        cute.make_layout(num_warps),  # thr_layout
        cute.make_layout(4),  # val_layout — 4 K indices per gather4
    )
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    warp_copy_AIdx_s2r = copy_AIdx_s2r.get_slice(warp_idx)
    tSR_sAIdx = warp_copy_AIdx_s2r.partition_S(sAIdx)  # (((4,1),4,4))
    # ((4,1),4,(64,2),(1,4)):((64,0),1024,(1,4096),(0,8192))
    tSR_sA = warp_copy_AIdx_s2r.partition_S(layout_utils.transpose_view(sA))
    tma_desc_ptr = get_tma_desc_addr(tma_atom)
    tma_gather4_load_fn = partial(tma_gather4_load, tma_desc_ptr, num_cta=cta_group)

    def prefetch_from_smem_fn(
        a_prefetch_pipeline,
        src_idx,
        dst_idx,
        a_prefetch_consumer_state,
    ) -> cute.Tensor:
        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
        tSR_rAIdx = load_s2r(tSR_sAIdx[None, None, dst_idx])
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
        return tSR_rAIdx

    def copy_fn(src_idx, dst_idx, tSR_rAIdx, tma_bar_ptr: cute.Pointer):
        # Issue gather4: col_idx = M position, row_indices = 4 K positions
        tSR_sA_cur = tSR_sA[None, None, None, dst_idx]
        gather_dim = cute.size(tSR_sA_cur, mode=[2, 0])  # Typically 64
        for k in cutlass.range(cute.size(tSR_rAIdx, mode=[1]), unroll_full=True):
            row_indices = [tSR_rAIdx[v, k] for v in range(4)]
            for m in cutlass.range(cute.size(tSR_sA_cur, mode=[2, 1]), unroll_full=True):
                smem_ptr = tSR_sA_cur[None, k, (None, m)].iterator
                with cute.arch.elect_one():
                    tma_gather4_load_fn(
                        smem_ptr, tma_bar_ptr, col_idx + m * gather_dim, row_indices
                    )

    return copy_fn, prefetch_from_smem_fn


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


@dsl_user_op
@cute.jit
def store(
    ptr: cute.Pointer,
    val,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Store a scalar value via cute.arch.store.

    ptr:  cute.Pointer (any address space).
    val:  DSL Numeric value.
    pred: None → unconditional.  DSL Boolean → skipped when pred == 0.
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def store_v2(
    ptr: cute.Pointer,
    v0,
    v1,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Vectorized store of 2 elements via cute.arch.store.

    Packs v0, v1 into an MLIR <2 x T> vector.
    ptr:  cute.Pointer (any address space, must be aligned for vector width).
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    vec = make_vector(type(v0), v0, v1, loc=loc, ip=ip)
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def store_v4(
    ptr: cute.Pointer,
    v0,
    v1,
    v2,
    v3,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Vectorized store of 4 elements via cute.arch.store.

    Packs v0–v3 into an MLIR <4 x T> vector.
    ptr:  cute.Pointer (any address space, must be aligned for vector width).
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    vec = make_vector(type(v0), v0, v1, v2, v3, loc=loc, ip=ip)
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
