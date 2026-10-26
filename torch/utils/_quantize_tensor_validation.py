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


def _validate_mxfp8_inputs(
    quant_format: _TensorQuantFormat,
    input: torch.Tensor,
    dim: list[int] | tuple[int, int],
    swizzle_type: int,
    square: bool,
    dual: bool,
) -> tuple[int, int]:
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_layout import (
        _build_blockscaled_tma_spec,
        _canonicalize_matrix_input,
        normalize_quantize_dim,
    )
    from torch.nn.functional import SwizzleType

    if quant_format != _TensorQuantFormat.MXFP8:
        raise AssertionError(f"Unsupported {quant_format=}")
    axes = normalize_quantize_dim(dim, input.ndim)
    swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4.value
    if dual:
        torch._check_value(
            swizzled,
            lambda: "dual mxfp8 quantization currently requires SWIZZLE_32_4_4",
        )
        torch._check_value(
            not square,
            lambda: "dual mxfp8 quantization does not yet support 32x32 MXFP8 scaling",
        )
        ordered = (min(axes), max(axes))
        kernel_dim = (ordered, (ordered[1], ordered[0]))
    else:
        torch._check_value(
            swizzle_type
            in (SwizzleType.NO_SWIZZLE.value, SwizzleType.SWIZZLE_32_4_4.value),
            lambda: "unsupported swizzle type",
        )
        kernel_dim = axes
    canonical = _canonicalize_matrix_input(input, kernel_dim, square, swizzled)
    _build_blockscaled_tma_spec(
        canonical.input,
        dim=canonical.dim,
        input_matrix_axes=canonical.input_matrix_axes,
        requested_dim=dim,
        is_square_scaling=square,
        is_scale_swizzled=swizzled,
    )
    return axes


def _quantize_tensor_validate_inputs(
    quant_format: _TensorQuantFormat,
    input: torch.Tensor,
    dim: list[int] | tuple[int, int],
    swizzle_type: int,
    scaling_type_use_square_block_size: bool,
) -> tuple[int, int]:
    return _validate_mxfp8_inputs(
        quant_format,
        input,
        dim,
        swizzle_type,
        scaling_type_use_square_block_size,
        False,
    )


def _quantize_tensor_dual_validate_inputs(
    quant_format: _TensorQuantFormat,
    input: torch.Tensor,
    dim: list[int] | tuple[int, int],
    swizzle_type: int,
    scaling_type_use_square_block_size: bool,
) -> tuple[int, int]:
    return _validate_mxfp8_inputs(
        quant_format, input, dim, swizzle_type, scaling_type_use_square_block_size, True
    )
