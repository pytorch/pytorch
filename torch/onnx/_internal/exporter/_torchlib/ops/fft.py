"""torch.ops.aten operators under the `fft` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC001, TC003

from __future__ import annotations

from collections.abc import Sequence

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


def _fftn_onnx_normalization(
    self: TFloat,
    normalization: int,
    signal_size,
    inverse: bool = False,
) -> TFloat:
    """Normalize in forward or backward direction."""
    # Norm values defined in https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOps.cpp#L117-L131
    # Norm modes: https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOpsUtils.h#L15-L19
    # Modes:
    # 0: no normalization (backward)
    # 1: "ortho" - divide by 1/sqrt(signal_size) (ortho)
    # 2: divide by signal_size (forward)
    signal_size = op.CastLike(signal_size, self)
    if not inverse:
        # Forward normalization
        if normalization == 1:
            self = op.Div(self, op.Sqrt(signal_size))
        elif normalization == 2:
            self = op.Div(self, signal_size)
    else:
        # Backward normalization, accounting for op.DFT already dividing by signal_size
        if normalization == 0:
            self = op.Mul(self, signal_size)
        elif normalization == 1:
            self = op.Mul(self, op.Sqrt(signal_size))
    return self


@onnx_impl(aten._fft_r2c.default, trace_only=True)
def aten__fft_r2c(
    self: TFloat, dim: Sequence[int], normalization: int, onesided: bool
) -> TFloat:
    """_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor

    Real to complex forward FFT.
    """

    # No need to fill the imaginary part because ONNX DFT accepts real inputs
    # https://onnx.ai/onnx/operators/onnx__DFT.html#inputs

    unsqueeze_first_dim = 0 in dim
    # 1. Add a new dimension for the end and batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = op.Unsqueeze(self, axes=[0, -1])
        dim = [d + 1 for d in dim]
    else:
        transformed = op.Unsqueeze(self, axes=[-1])

    for idx, dimension in enumerate(reversed(dim)):
        # Explicitly pass the dft_length as the size of the transformed signal
        # along the target axis. Without this, some ONNX opset converters
        # (e.g. when converting DFT-17, where dft_length is an optional
        # input, to later opsets where it is required) may synthesize an
        # incorrect default dft_length, silently truncating the output to a
        # single frequency bin instead of inferring the length from the
        # input shape. See https://github.com/pytorch/pytorch/issues/155997
        dft_length = op.Squeeze(
            op.Shape(transformed, start=dimension, end=dimension + 1), axes=[0]
        )
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            dft_length,
            inverse=False,
        )
        if idx > 0:
            transformed = op.DFT(
                transformed, dft_length, axis=dimension, inverse=False, onesided=False
            )
        else:
            # Torch computes one-sided FFT on the last dimension only.
            transformed = op.DFT(
                transformed,
                dft_length,
                axis=dimension,
                inverse=False,
                onesided=onesided,
            )

    if unsqueeze_first_dim:
        transformed = op.Squeeze(transformed, axes=[0])

    return transformed
