# Copyright (c) 2025, Tri Dao.
"""Epilogue utilities: shared helpers for epilogue mixin classes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils

from . import sm90_utils
from . import copy_utils


def assume_stride_divisibility(tensor):
    """Assume all strides are divisible by 32 bits (except static strides).

    Used for broadcast vectors and similar tensors where stride alignment is guaranteed.
    Returns a new tensor with the assumed strides.
    """
    if tensor is None:
        return None
    new_stride = tuple(
        cute.assume(s, divby=32 // tensor.element_type.width) if not cute.is_static(s) else s
        for s in tensor.stride
    )
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride))


def assume_broadcast_strides(*tensors):
    """Apply stride divisibility assumptions to multiple broadcast vectors.

    Returns a list with None preserved for None inputs.
    """
    return [assume_stride_divisibility(t) for t in tensors]


def setup_epi_tensor(gemm, tensor, epi_tile=None, op_type="store"):
    """Create TMA atom + smem layout for a supplemental epilogue tensor.

    Args:
        gemm: The GEMM object (provides arch, epi_stage, _make_tma_epi_atoms_and_tensors).
        tensor: The global memory tensor to set up TMA for.
        epi_tile: Epilogue tile shape. Defaults to gemm.epi_tile.
        op_type: "store" or "load".

    Returns:
        (tma_atom, tma_tensor, smem_layout_staged, epi_tile)
    """
    if epi_tile is None:
        epi_tile = gemm.epi_tile
    dtype = tensor.element_type
    layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
    utils_cls = sm100_utils if gemm.arch >= 100 else sm90_utils
    smem_layout_staged = utils_cls.make_smem_layout_epi(dtype, layout, epi_tile, gemm.epi_stage)
    tma_input = (
        copy_utils.create_ragged_tensor_for_tma(tensor, ragged_dim=0, ptr_shift=True)
        if cute.rank(tensor) == 2
        else tensor
    )
    tma_atom, tma_tensor = gemm._make_tma_epi_atoms_and_tensors(
        tma_input,
        smem_layout_staged,
        epi_tile,
        op_type=op_type,
    )
    return tma_atom, tma_tensor, smem_layout_staged, epi_tile
