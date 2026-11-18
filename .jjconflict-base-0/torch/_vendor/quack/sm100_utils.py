# Copyright (c) 2025, Tri Dao.

from typing import Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_og
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05


class MmaMXF4NVF4Op(tcgen05.MmaMXF4NVF4Op):
    """tcgen05 kind::mxf4nvf4 MMA op with the scale config as a parameter.

    kind::mxf4nvf4 is ONE MMA atom covering both both-fp4 scale configs -
    scale_vec::2X (vec 32, e8m0; the mxfp4 instantiation, which PTX also
    spells ``kind::mxf4``) and scale_vec::4X (vec 16; nvfp4) - mirroring
    CUTLASS C++'s ``SM100_MMA_MXF4_SS<..., VS>``. The DSL class pins
    sf_vec_size = 16; this subclass restores the parameter. Both vec sizes
    build the identical MLIR atom type (it has no kind attribute - the
    backend derives the PTX spelling from the vec size), so this changes
    which Python op models the atom, not the lowered instruction.
    """

    def __init__(self, sf_dtype, sf_vec_size, instruction_shape, cta_group, a_src):
        tcgen05.BlockScaledMmaOp.__init__(
            self,
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float32,
            sf_dtype,
            sf_vec_size,
            instruction_shape,
            cta_group,
            a_src,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        self._verify()


def make_blockscaled_trivial_tiled_mma(
    a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, mma_tiler_mn
) -> cute.TiledMma:
    """Like the DSL helper, but both-fp4 pairs always run the single
    kind::mxf4nvf4 atom with the format's scale config (the DSL helper splits
    them into MmaMXF4Op / MmaMXF4NVF4Op by vec size)."""
    if a_dtype is cutlass.Float4E2M1FN and b_dtype is cutlass.Float4E2M1FN:
        op = MmaMXF4NVF4Op(
            sf_dtype, sf_vec_size, (*mma_tiler_mn, 64), cta_group, tcgen05.OperandSource.SMEM
        )
        return cute.make_tiled_mma(cute.make_mma_atom(op))
    return sm100_utils_og.make_blockscaled_trivial_tiled_mma(
        a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, mma_tiler_mn
    )


@dsl_user_op
def make_smem_layout_cpasync_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """
    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode,
            a_dtype,
            a_smem_shape_mn_k,
            loc=loc,
            ip=ip,
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged


@dsl_user_op
def make_smem_layout_atom_tma_gather_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    gather_size: int = 4,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """SMEM load layout atom for A with TMA gather4."""
    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    # e,g., S<3, 4, 3> o  0 o (8, 64):(64, 1) for k_major
    # e,g., S<3, 4, 3> o  0 o (64, 8):(1, 64) for m_major
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode, a_dtype, a_smem_shape_mn_k, loc=loc, ip=ip
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    swizzle = a_smem_layout_atom.inner
    smem_layout = a_smem_layout_atom.outer
    if is_k_major:
        # Replace M-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (gather_size, smem_layout.shape[1]), stride=smem_layout.stride, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
    else:
        # Replace K-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (smem_layout.shape[0], gather_size), stride=smem_layout.stride, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
    return a_smem_layout_atom


@dsl_user_op
def make_smem_layout_tma_gather_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """SMEM load layout for A with TMA gather4."""
    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = make_smem_layout_atom_tma_gather_a(
        tiled_mma, mma_tiler_mnk, a_dtype, loc=loc, ip=ip
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=(1, 0, 2) if not is_k_major else (0, 1, 2),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged
