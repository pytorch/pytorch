# Copyright (c) 2025, Tri Dao.
# Ported Cutlass code from C++ to Python:
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp

from enum import IntEnum

import cutlass
import cutlass.cute as cute

# ---------------------------------------------------------------------------
# Enumerations that match the HW encodings (values MUST stay identical)
# ---------------------------------------------------------------------------


class Major(IntEnum):  # matrix “layout” in the ISA docs
    K = 0
    MN = 1


class ScaleIn(IntEnum):  # negate flags
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):  # 2-bit field (bits 4-5)
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):  # 3-bit field (A/B element type)
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


# ---------------------------------------------------------------------------
# CUTLASS-type → encoding helpers
# ---------------------------------------------------------------------------


def to_UMMA_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 3-bit encoding for Matrix A/B.
    """
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    # Unsigned 8-bit (if available in your CUTLASS build)
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    # FP-16 / BF-16
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    # TensorFloat-32 (8-bit exponent, 10-bit mantissa packed in 19 bits)
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    # Float-8 / Float-6 / Float-4 – add whenever CUTLASS exposes them
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 2-bit accumulator encoding.
    """
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}")


# ---------------------------------------------------------------------------
# The constructor – accepts only CUTLASS scalar classes
# ---------------------------------------------------------------------------


def make_instr_desc(
    a_type,  # CUTLASS scalar class, e.g. cutlass.Int8
    b_type,
    c_type,
    M: int,  # 64, 128 or 256
    N: int,  # 8 … 256 (multiple of 8)
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    """
    Build the 32-bit instruction descriptor for Blackwell MMA.
    All matrix/accumulator **types must be CUTLASS scalar classes** –
    passing integers is forbidden.
    """
    # --- encode element formats -------------------------------------------------
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    # --- range checks on M/N -----------------------------------------------------
    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4  # 5-bit field
    n_dim = N >> 3  # 6-bit field

    # fmt: off
    # --- pack the bit-fields -----------------------------------------------------
    desc = 0
    desc |= (0                 & 0x3) << 0        # sparse_id2 (always 0 here)
    desc |= (int(is_sparse)    & 0x1) << 2        # sparse_flag
    desc |= (int(c_sat)        & 0x1) << 3        # saturate
    desc |= (c_fmt             & 0x3) << 4        # c_format
    desc |= (a_fmt             & 0x7) << 7        # a_format
    desc |= (b_fmt             & 0x7) << 10       # b_format
    desc |= (int(a_neg)        & 0x1) << 13       # a_negate
    desc |= (int(b_neg)        & 0x1) << 14       # b_negate
    desc |= (int(a_major)      & 0x1) << 15       # a_major
    desc |= (int(b_major)      & 0x1) << 16       # b_major
    desc |= (n_dim             & 0x3F) << 17      # n_dim (6 bits)
    desc |= (m_dim             & 0x1F) << 24      # m_dim (5 bits)
    desc |= (int(max_shift)    & 0x3) << 30       # max_shift (2 bits)
    # fmt: on

    return desc & 0xFFFF_FFFF  # ensure 32-bit result


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
        Major.K if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
    )


class LayoutType(IntEnum):  # occupies the top-3 bits [61:64)
    SWIZZLE_NONE = 0  # (a.k.a. “INTERLEAVE” in older docs)
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6
    # values 3,5,7 are reserved / illegal for UMMA


# ---------------------------------------------------------------------------
#  Helpers – figure out the SWIZZLE_* family from the tensor layout
# ---------------------------------------------------------------------------


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:  # Swizzle<*,4,3>
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[B]  # KeyError ⇒ invalid B→ raise
    if M == 5:  # Swizzle<2,5,2> (the only legal triple for M==5)
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    # Any other (M,B,S) triple is not a UMMA-legal shared-memory layout
    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(layout: cute.Layout, swizzle: cute.Swizzle, major: Major) -> int:
    """
    Convert a 2-D *shared-memory* Cute layout into the Blackwell 64-bit
    smem-descriptor, without the smem start address.
    layout must correspond to layout of an uint128 tensor.
    """
    # ------------------------------------------------------------------ meta
    layout_type = _layout_type(swizzle)  # resolve SWIZZLE_* family

    VERSION = 1  # bits 46–47
    LBO_MODE = 0  # bit  52
    BASE_OFFSET = 0  # bits 49–51   (CUTLASS always 0)

    # ---------------------------------------------------------- strides  (units: uint128_t = 16 B)
    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(layout, (swizzle_atom_mn_size, swizzle_atom_k_size))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_MN Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = canonical_layout.stride[0][1], canonical_layout.stride[1][1]
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError("Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.")
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    # ------------------------------------------------------------------ pack
    desc = 0
    # leading_byte_offset_  [16:30)
    desc |= (leading_byte_offset & 0x3FFF) << 16
    # stride_byte_offset_   [32:46)
    desc |= (stride_byte_offset & 0x3FFF) << 32
    # version_             [46:48)
    desc |= (VERSION & 0x3) << 46
    # base_offset_         [49:52)
    desc |= (BASE_OFFSET & 0x7) << 49
    # lbo_mode_            [52:53)
    desc |= (LBO_MODE & 0x1) << 52
    # layout_type_         [61:64)
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF  # force 64-bit width


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    # 14 bits, remove 4 LSB (bits 0-13 in desc)
    return (start_addr.toint() & 0x3FFFF) >> 4


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    sA_swizzle = sA.iterator.type.swizzle_type
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        sA_swizzle,
        major,
    )
