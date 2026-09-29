"""CuTeDSL implementations of the MXFP8 quantization operators."""

import torch

from ... import cutedsl_utils as cu


def _quantize_tensor_impl(
    input: torch.Tensor,
    qdata_dtype: torch.dtype,
    inner_scale_calc: int,
    scaling_type: int,
    swizzle_type: int,
    scaling_type_square_block_and_expand: bool = False,
) -> list[torch.Tensor]:
    if qdata_dtype != torch.float8_e4m3fn or inner_scale_calc != 0 or scaling_type != 3:
        raise ValueError("only MXFP8 RCEIL with BlockWise1x32 is supported")
    if swizzle_type not in (0, 1):
        raise ValueError("unsupported swizzle type")
    if input.dim() != 2:
        raise ValueError("quantize_tensor requires a 2D input")
    if input.is_contiguous():
        source, orientation = input, "dim_k"
    else:
        source = input.t()
        if not source.is_contiguous():
            raise ValueError("input must be contiguous or a transpose of contiguous")
        orientation = "dim_m"
    if orientation == "dim_m" and swizzle_type != 1:
        raise ValueError("dim-m quantization requires SWIZZLE_32_4_4")
    if scaling_type_square_block_and_expand and (
        orientation != "dim_k" or swizzle_type != 1
    ):
        raise ValueError("32x32 MXFP8 scaling requires dim-k and SWIZZLE_32_4_4")
    if input.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("quantize_tensor does not support autograd")
    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("quantize_tensor requires an NVIDIA CUDA tensor")
    if torch.cuda.get_device_capability(input.device) < (10, 0):
        raise RuntimeError("quantize_tensor requires CUDA capability 10.0 or newer")

    from .blockscaled_tma.blockscaled_tma_impl import _blockscaled_tma_impl

    return list(
        _blockscaled_tma_impl(
            source,
            orientation,
            scaling_type_square_block_and_expand,
            swizzle_type == 1,
        )
    )


def _quantize_tensor_dual_impl(
    input: torch.Tensor,
    qdata_dtype: torch.dtype,
    inner_scale_calc: int,
    scaling_type: int,
    swizzle_type: int,
    scaling_type_square_block_and_expand: bool = False,
) -> list[torch.Tensor]:
    if qdata_dtype != torch.float8_e4m3fn or inner_scale_calc != 0 or scaling_type != 3:
        raise ValueError("only MXFP8 RCEIL with BlockWise1x32 is supported")
    if swizzle_type != 1:
        raise ValueError("dual quantization requires SWIZZLE_32_4_4")
    if scaling_type_square_block_and_expand:
        raise ValueError("dual quantization does not support 32x32 MXFP8 scaling")
    if input.dim() != 2:
        raise ValueError("quantize_tensor_dual requires a 2D input")
    if not input.is_contiguous():
        raise ValueError("quantize_tensor_dual requires a contiguous input")
    if input.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("quantize_tensor_dual does not support autograd")
    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("quantize_tensor_dual requires an NVIDIA CUDA tensor")
    if torch.cuda.get_device_capability(input.device) < (10, 0):
        raise RuntimeError("dual quantization requires CUDA capability 10.0 or newer")

    from .blockscaled_tma.blockscaled_tma_impl import _blockscaled_tma_impl

    return list(_blockscaled_tma_impl(input, "dim_km", False, True))


def register_to_dispatch() -> None:
    cu.register_op_override(
        "aten",
        "_quantize_tensor",
        "CUDA",
        cond=None,
        impl=_quantize_tensor_impl,
        unconditional_override=True,
    )
    cu.register_op_override(
        "aten",
        "_quantize_tensor_dual",
        "CUDA",
        cond=None,
        impl=_quantize_tensor_dual_impl,
        unconditional_override=True,
    )
