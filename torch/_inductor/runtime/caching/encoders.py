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


def get_matmul_precision_for_cache(mat: Tensor) -> str:
    """Return the device-specific FP32 policy used to key matmul decisions."""
    # Non-FP32 matmuls ignore this policy and share one stable cache value.
    if mat.dtype != torch.float32:
        return "not_float32"
    if mat.device.type == "cuda":
        return torch.backends.cuda.matmul.fp32_precision
    return torch.backends.mkldnn.fp32_precision


class ShouldPadEncodedParams(TypedDict):
    """TypedDict for encoded should_pad parameters."""

    mat1: EncodedTensor
    mat2: EncodedTensor
    op: str
    input: EncodedTensor | None
    mat1_exclude_padding_time: bool
    mat2_exclude_padding_time: bool
    fp32_precision: str
    padding_plan_version: int
    addmm_scalars: tuple[float, float] | None
    selection_policy: tuple[object, float, bool, bool]
    device: tuple[str, str | None, str | None]


def get_device_identity(device: torch.device) -> tuple[str, str | None, str | None]:
    """Return the logical device plus the physical model and architecture."""
    name: str | None = None
    arch: str | None = None
    try:
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            name = properties.name
            if torch.version.hip is not None:
                arch = properties.gcnArchName.split(":", 1)[0]
            else:
                arch = f"sm_{properties.major}{properties.minor}"
        elif device.type == "xpu":
            properties = torch.xpu.get_device_properties(device)
            name = properties.name
            xpu_arch = getattr(properties, "architecture", None)
            arch = str(xpu_arch) if xpu_arch is not None else None
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        # Fake accelerator tensors can be used on hosts without that backend.
        pass
    return (str(device), name, arch)


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
    from torch._inductor.fx_passes.pad_mm import (
        padding_selection_policy,
        should_exclude_padding_time,
    )

    return ShouldPadEncodedParams(
        mat1=_encode_tensor(mat1),
        mat2=_encode_tensor(mat2),
        op=str(op),
        input=_encode_tensor(input) if input is not None else None,
        mat1_exclude_padding_time=should_exclude_padding_time(match, "mat1"),
        mat2_exclude_padding_time=should_exclude_padding_time(match, "mat2"),
        fp32_precision=get_matmul_precision_for_cache(mat1),
        padding_plan_version=3,
        addmm_scalars=(
            (match.kwargs.get("beta", 1.0), match.kwargs.get("alpha", 1.0))
            if op is torch.ops.aten.addmm
            else None
        ),
        selection_policy=padding_selection_policy(),
        device=get_device_identity(mat1.device),
    )
