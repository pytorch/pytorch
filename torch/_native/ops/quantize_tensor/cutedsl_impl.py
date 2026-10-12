"""CuTeDSL implementations of the MXFP8 quantization operators."""

import torch
from torch._quantization_formats import (
    _get_quant_format,
    _quantize_tensor_dual_validate_inputs,
    _quantize_tensor_validate_inputs,
    _TensorQuantFormat,
)

from ... import cutedsl_utils as cu
from .utils import _device_capability


def _quantize_tensor_impl(
    input: torch.Tensor,
    scaling_type: int,
    qdata_dtype: torch.dtype,
    scaling_algorithm: int,
    swizzle_type: int,
    scaling_type_use_square_block_size: bool = False,
) -> list[torch.Tensor]:
    from torch.nn.functional import SwizzleType

    if input.dim() != 2:
        raise ValueError("quantize_tensor requires a 2D input")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("quantize_tensor supports only fp16, bf16, and fp32 input")

    quant_format = _get_quant_format(qdata_dtype, scaling_algorithm, scaling_type)
    if quant_format == _TensorQuantFormat.MXFP8:
        _quantize_tensor_validate_inputs(
            quant_format,
            input,
            scaling_type,
            qdata_dtype,
            scaling_algorithm,
            swizzle_type,
            scaling_type_use_square_block_size,
        )
    else:
        raise AssertionError(f"Unknown {quant_format=}")

    is_scale_swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4.value
    if input.is_contiguous():
        source, orientation = input, "dim_k"
    else:
        source, orientation = input.t(), "dim_m"
    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("quantize_tensor requires an NVIDIA CUDA tensor")
    if _device_capability(input.get_device()) < (10, 0):
        raise RuntimeError("quantize_tensor requires CUDA capability 10.0 or newer")
    if source.data_ptr() % 16 != 0:
        raise ValueError("quantize_tensor requires a 16-byte-aligned input")

    from .blockscaled_tma.blockscaled_tma_impl import _blockscaled_tma_impl

    return list(
        _blockscaled_tma_impl(
            source,
            orientation,
            scaling_type_use_square_block_size,
            is_scale_swizzled,
        )
    )


def _quantize_tensor_dual_impl(
    input: torch.Tensor,
    scaling_type: int,
    qdata_dtype: torch.dtype,
    scaling_algorithm: int,
    swizzle_type: int,
    scaling_type_use_square_block_size: bool = False,
) -> list[torch.Tensor]:
    if input.dim() != 2:
        raise ValueError("quantize_tensor_dual requires a 2D input")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dual quantization supports only fp16, bf16, and fp32 input")

    quant_format = _get_quant_format(qdata_dtype, scaling_algorithm, scaling_type)
    if quant_format == _TensorQuantFormat.MXFP8:
        _quantize_tensor_dual_validate_inputs(
            quant_format,
            input,
            scaling_type,
            qdata_dtype,
            scaling_algorithm,
            swizzle_type,
            scaling_type_use_square_block_size,
        )
    else:
        raise AssertionError(f"Unknown {quant_format=}")

    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("quantize_tensor_dual requires an NVIDIA CUDA tensor")
    if _device_capability(input.get_device()) < (10, 0):
        raise RuntimeError("dual quantization requires CUDA capability 10.0 or newer")
    if input.data_ptr() % 16 != 0:
        raise ValueError("quantize_tensor_dual requires a 16-byte-aligned input")

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
