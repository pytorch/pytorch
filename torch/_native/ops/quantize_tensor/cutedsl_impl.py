"""CuTeDSL implementations of the MXFP8 quantization operators."""

import torch

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
    from torch.nn.functional import ScalingAlgorithm, ScalingType, SwizzleType

    if input.dim() != 2:
        raise ValueError("quantize_tensor requires a 2D input")
    if input.size(1) % 32 != 0:
        raise ValueError("quantize_tensor requires columns divisible by 32")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("quantize_tensor supports only fp16, bf16, and fp32 input")
    if qdata_dtype != torch.float8_e4m3fn:
        raise ValueError("quantize_tensor supports only float8_e4m3fn qdata")
    if scaling_algorithm != ScalingAlgorithm.MXFP_E8M0_RU.value:
        raise ValueError("quantize_tensor requires scaling_algorithm=MXFP_E8M0_RU")
    if scaling_type != ScalingType.BlockWise1x32.value:
        raise ValueError("quantize_tensor supports only BlockWise1x32 scaling")
    if swizzle_type not in (
        SwizzleType.NO_SWIZZLE.value,
        SwizzleType.SWIZZLE_32_4_4.value,
    ):
        raise ValueError("unsupported quantize_tensor swizzle type")
    is_scale_swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4.value
    if input.is_contiguous():
        source, orientation = input, "dim_k"
    else:
        source = input.t()
        if not source.is_contiguous():
            raise ValueError("input must be contiguous or a transpose of contiguous")
        orientation = "dim_m"
    if orientation == "dim_m" and not is_scale_swizzled:
        raise ValueError("dim-m quantization requires SWIZZLE_32_4_4")
    if scaling_type_use_square_block_size and (
        orientation != "dim_k" or not is_scale_swizzled
    ):
        raise ValueError("32x32 MXFP8 scaling requires dim-k and SWIZZLE_32_4_4")
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
    from torch.nn.functional import ScalingAlgorithm, ScalingType, SwizzleType

    if input.dim() != 2:
        raise ValueError("quantize_tensor_dual requires a 2D input")
    if input.size(0) % 32 != 0 or input.size(1) % 32 != 0:
        raise ValueError("dual quantization requires both dimensions divisible by 32")
    if not input.is_contiguous():
        raise ValueError("quantize_tensor_dual requires a contiguous input")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dual quantization supports only fp16, bf16, and fp32 input")
    if qdata_dtype != torch.float8_e4m3fn:
        raise ValueError("quantize_tensor_dual supports only float8_e4m3fn qdata")
    if scaling_algorithm != ScalingAlgorithm.MXFP_E8M0_RU.value:
        raise ValueError("quantize_tensor_dual requires scaling_algorithm=MXFP_E8M0_RU")
    if scaling_type != ScalingType.BlockWise1x32.value:
        raise ValueError("quantize_tensor_dual supports only BlockWise1x32 scaling")
    if swizzle_type != SwizzleType.SWIZZLE_32_4_4.value:
        raise ValueError("quantize_tensor_dual requires SWIZZLE_32_4_4")
    if scaling_type_use_square_block_size:
        raise ValueError("quantize_tensor_dual does not support 32x32 MXFP8 scaling")
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
