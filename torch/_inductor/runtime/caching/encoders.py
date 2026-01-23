"""
Custom encoder functions

This module provides reusable encoder functions that convert function parameters
into JSON-serializable dictionaries for caching purposes.
"""

from typing_extensions import TypedDict

import torch
from torch import Tensor
from torch._inductor.pattern_matcher import Match
from torch._inductor.runtime.caching.utils import _encode_tensor, EncodedTensor


class ShouldPadEncodedParams(TypedDict):
    """TypedDict for encoded should_pad parameters."""

    mat1: EncodedTensor
    mat2: EncodedTensor
    op: str
    input: EncodedTensor | None
    mat1_exclude_padding_time: bool
    mat2_exclude_padding_time: bool
    tf32: bool


def should_pad_params_encoder(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> ShouldPadEncodedParams:
    """Encode parameters for _should_pad into a human-readable dict.

    This encoder extracts only the information needed for caching:
    - Tensor shape, stride, and dtype (not the actual data)
    - Whether padding time should be excluded for mat1 and mat2
    - The operation as a string

    Args:
        match: The pattern match object
        mat1: First matrix tensor
        mat2: Second matrix tensor
        op: The operation being performed
        input: Optional input tensor for addmm

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    # Import here to avoid circular dependency
    from torch._inductor.fx_passes.pad_mm import should_exclude_padding_time

    return ShouldPadEncodedParams(
        mat1=_encode_tensor(mat1),
        mat2=_encode_tensor(mat2),
        op=str(op),
        input=_encode_tensor(input) if input is not None else None,
        mat1_exclude_padding_time=should_exclude_padding_time(match, "mat1"),
        mat2_exclude_padding_time=should_exclude_padding_time(match, "mat2"),
        tf32=False
        if mat1.dtype != torch.float32
        else bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.mkldnn.allow_tf32
        ),
    )
