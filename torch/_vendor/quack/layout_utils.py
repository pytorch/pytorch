# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.


import cutlass
import cutlass.cute as cute

from cutlass import Int32, const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def concat_to_interleave(a: cute.Tensor, dim: int) -> cute.Tensor:
    """Reshape a concat [first_half; second_half] layout to interleaved along `dim`.

    Splits dimension `dim` (size 2N) into hierarchical (2, N) so that elements
    from the first half and second half alternate: [first_0, second_0, first_1, ...].
    Used to convert gated MLP weight layout from concat [gate; up] to interleaved.
    """
    half = cute.size(a, mode=[dim]) // 2
    shape = (*a.shape[:dim], (2, half), *a.shape[dim + 1 :])
    stride = (*a.stride[:dim], (half * a.stride[dim], a.stride[dim]), *a.stride[dim + 1 :])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
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
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = cute.arch.prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = cute.arch.prmt(upper0, lower0, selector_lower)


@cute.jit
def permute_Cregs_b32_for_stsm(t: cute.Tensor) -> None:
    """Permute and shuffle within 4 threads to change the layout from
     T0 | T1  | T2  | T3
    a b | c d | e f | g h
    to
    T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3
    a  | b  | c  | d  | e  | f  | g  | h
    This is so that we can use STSM (instead of STS.64) to store C registers without bank conflict.
    """

    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    # left_map = [0, 2, 1, 3]
    # right_map = [2, 0, 3, 1]
    # indexing isn't supported so we have to do arithmetic
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b10

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            # a b | c d | e f | g h -> a b | c d | f e | h g
            left0 = left if quad_idx < 2 else right
            right0 = right if quad_idx < 2 else left
            # a b | c d | f e | h g -> a b | f d | c e | h g
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            # a b | f d | c e | h g -> a e | f b | c g | h d
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            # a e | f b | c g | h d -> a e | b f | c g | d h
            t[i * 4 + r * 2 + 0] = left0 if quad_idx % 2 == 0 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx % 2 == 0 else left0
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]


@cute.jit
def permute_Cregs_b32_for_ldsm(t: cute.Tensor) -> None:
    """Permute and shuffle within 4 threads to change the layout from
    T0 | T1 | T2 | T3 | T0 | T1 | T2 | T3
    a  | b  | c  | d  | e  | f  | g  | h
    to
     T0 | T1  | T2  | T3
    a b | c d | e f | g h
    This is so that we can use LDSM (instead of LDS.64) to store C registers without bank conflict.
    """

    assert t.element_type.width == 32
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b32 permutation"

    quad_idx = cute.arch.lane_idx() % 4
    # left_map = [0, 2, 1, 3]
    # right_map = [1, 3, 0, 2]
    # indexing isn't supported so we have to do arithmetic
    left_idx = quad_idx // 2 if quad_idx % 2 == 0 else 2 + quad_idx // 2
    right_idx = left_idx ^ 0b01

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    # This is just the inverse of permute_Cregs_b32_for_stsm
    for i in cutlass.range(cute.size(t.shape) // 4, unroll_full=True):
        t[i * 4 + 1], t[i * 4 + 2] = t[i * 4 + 2], t[i * 4 + 1]
        for r in cutlass.range(2, unroll_full=True):
            left, right = t[i * 4 + r * 2 + 0], t[i * 4 + r * 2 + 1]
            # a e | b f | c g | d h -> a e | f b | c g | h d
            left0 = left if quad_idx % 2 == 0 else right
            right0 = right if quad_idx % 2 == 0 else left
            # a e | f b | c g | h d -> a b | f d | c e | h g
            right0 = cute.arch.shuffle_sync(right0, offset=right_idx, mask_and_clamp=mask_and_clamp)
            # a b | f d | c e | h g -> a b | c d | f e | h g
            left0 = cute.arch.shuffle_sync(left0, offset=left_idx, mask_and_clamp=mask_and_clamp)
            # a b | c d | f e | h g -> a b | c d | e f | g h
            t[i * 4 + r * 2 + 0] = left0 if quad_idx < 2 else right0
            t[i * 4 + r * 2 + 1] = right0 if quad_idx < 2 else left0


@cute.jit
def concat_layout(*layouts: cute.Layout) -> cute.Layout:
    return cute.make_layout(
        tuple(l.shape for l in layouts),
        stride=tuple(l.stride for l in layouts),
    )


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # If N / 8 is odd, we'll convert to ((2, 2, 1), MMA_M, N / 8, MMA_N).
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(
            acc_layout, ((None, None, div), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        assert acc_layout.shape[2] % 2 == 0
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def convert_layout_zero_stride(
    input: cute.Tensor | cute.Layout, ref_layout: cute.Layout
) -> cute.Layout:
    layout = input.layout if const_expr(isinstance(input, cute.Tensor)) else input
    # Group the modes with non-zero stride in the ref_layout together,
    # and the modes with zero stride together
    layout_flat = cute.flatten(layout)
    ref_layout_flat = cute.flatten(ref_layout)
    nonzero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride != 0]
    zero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride == 0]
    # There's an edge case when all modes are zero stride
    new_shape = (
        tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    out_layout = cute.make_layout(new_shape, stride=new_stride)
    if const_expr(isinstance(input, cute.Tensor)):
        return cute.make_tensor(input.iterator, out_layout)
    else:
        return out_layout


def mma_partition_C_vec(
    sVec: cute.Tensor, thr_mma: cute.core.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def mma_partition_A_vec(
    sVec: cute.Tensor, thr_mma: cute.core.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_A(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_S_vec(
    sVec: cute.Tensor, thr_copy: cute.core.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_S(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def copy_partition_D_vec(
    sVec: cute.Tensor, thr_copy: cute.core.ThrCopy, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_thr = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = reshape_acc_to_mn(thr_copy.partition_D(sVec_thr))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


def tile_atom_to_shape_SF_strided(
    shape: cute.Shape,
    sf_vec_size: int,
    sf_strides,
) -> cute.Layout:
    """Build an SFA/SFB layout matching `shape` (A or B operand shape) but
    honoring the scale tensor's actual strides instead of hardcoded packed
    ones.

    Mirrors `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF(shape,
    sf_vec_size)`, except outer-mode strides come from `sf_strides` (pass
    `mSFA.stride` / `mSFB.stride` directly). The inner 512-B atom
    `((32, 4), (sf_vec_size, 4)) : ((16, 4), (0, 1))` is hardware-fixed.

    Implementation uses `cute.blocked_product(atom, outer)`; `blocked_product`
    scales the outer layout's strides by `cosize(atom) == 512`, so we divide
    the byte strides by 512 (one tile) before handing them in.

    Args:
        shape: A/B operand shape. Rank-3 `(m/n, k, l)` or rank-2
            `(total_mn, k)` (varlen_m).
        sf_vec_size: Scale factor vector size (16 or 32).
        sf_strides: Strides of the scale tensor, which has logical shape
            `(L, rmn, rk, 512)` (rank 4). Only `sf_strides[0..2]` are used:
            `sf_strides[1]` as the rmn stride, `sf_strides[2]` as the rk
            stride, and `sf_strides[0]` as the L stride (only for rank-3
            `shape`).
    """
    from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk

    atom = BlockScaledBasicChunk(sf_vec_size).layout
    rmn = cute.ceil_div(shape[0], 128)
    rk = cute.ceil_div(shape[1], sf_vec_size * 4)
    outer = cute.make_layout((rmn, rk), stride=(sf_strides[1] // 512, sf_strides[2] // 512))
    sf_layout = cute.blocked_product(atom, outer)
    if const_expr(len(shape) == 3):
        sf_layout = cute.append(sf_layout, cute.make_layout(shape[2], stride=sf_strides[0]))
    return sf_layout
