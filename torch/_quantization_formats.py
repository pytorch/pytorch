from enum import Enum

import torch


class _TensorQuantFormat(Enum):
    # A private enum to help keep quantization validation sane
    # TODO(future PR): also use this for F.scaled_mm and F.scaled_grouped_mm
    MXFP8 = "mxfp8"


def _get_quant_format(
    qdata_dtype: torch.dtype, scaling_algorithm: int, scaling_type: int
) -> _TensorQuantFormat:
    from torch.nn.functional import ScalingAlgorithm, ScalingType

    if (
        qdata_dtype == torch.float8_e4m3fn
        and scaling_algorithm == ScalingAlgorithm.MXFP_E8M0_RU.value
        and scaling_type == ScalingType.BlockWise1x32.value
    ):
        return _TensorQuantFormat.MXFP8
    raise ValueError(
        "unsupported quantization format for "
        f"qdata_dtype={qdata_dtype}, scaling_algorithm={scaling_algorithm}, "
        f"scaling_type={scaling_type}"
    )


def _quantize_tensor_validate_inputs(
    quant_format,
    input: torch.Tensor,
    scaling_type: int,
    qdata_dtype: torch.dtype,
    scaling_algorithm: int,
    swizzle_type: int,
    scaling_type_use_square_block_size: bool,
) -> None:
    # Input validation for `F.quantize_tensor`, shared between meta functions
    # and DSL implementations

    from torch.nn.functional import SwizzleType

    if quant_format == _TensorQuantFormat.MXFP8:
        rows, cols = input.shape
        torch._check_value(
            cols % 32 == 0,
            lambda: "quantize_tensor mxfp8 requires columns divisible by 32",
        )
        is_dim_k = input.is_contiguous()
        if not is_dim_k:
            torch._check_value(
                input.t().is_contiguous(),
                lambda: "input must be contiguous or a transpose of contiguous",
            )
        swizzled_value = SwizzleType.SWIZZLE_32_4_4.value
        torch._check_value(
            swizzle_type in (SwizzleType.NO_SWIZZLE.value, swizzled_value),
            lambda: "unsupported swizzle type",
        )
        is_scale_swizzled = swizzle_type == swizzled_value
        torch._check_value(
            is_dim_k or is_scale_swizzled,
            lambda: "dim-m mxfp8 quantization currently requires SWIZZLE_32_4_4",
        )
        torch._check_value(
            not scaling_type_use_square_block_size or (is_dim_k and is_scale_swizzled),
            lambda: "32x32 mxfp8 scaling currently requires dim-k and SWIZZLE_32_4_4",
        )
        if not is_dim_k:
            torch._check_value(rows % 16 == 0)
        if scaling_type_use_square_block_size:
            torch._check_value(rows % 32 == 0)

    else:
        raise AssertionError(f"Unsupported {quant_format=}")


def _quantize_tensor_dual_validate_inputs(
    quant_format,
    input: torch.Tensor,
    scaling_type: int,
    qdata_dtype: torch.dtype,
    scaling_algorithm: int,
    swizzle_type: int,
    scaling_type_use_square_block_size: bool,
) -> None:
    # Input validation for `F.quantize_tensor_dual`, shared between meta functions
    # and DSL implementations

    from torch.nn.functional import SwizzleType

    if quant_format == _TensorQuantFormat.MXFP8:
        rows, cols = input.shape
        torch._check_value(
            rows % 32 == 0,
            lambda: "dual mxfp8 quantization requires both dimensions divisible by 32",
        )
        torch._check_value(
            cols % 32 == 0,
            lambda: "dual mxfp8 quantization requires both dimensions divisible by 32",
        )
        torch._check_value(
            input.is_contiguous(),
            lambda: "dual mxfp8 quantization requires contiguous input",
        )
        torch._check_value(
            swizzle_type == SwizzleType.SWIZZLE_32_4_4.value,
            lambda: "dual mxfp8 quantization currently requires SWIZZLE_32_4_4",
        )
        torch._check_value(
            not scaling_type_use_square_block_size,
            lambda: "dual mxfp8 quantization does not yet support 32x32 MXFP8 scaling",
        )
    else:
        raise AssertionError(f"Unsupported {quant_format=}")
