# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR fft translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx_ir as ir


aten = torch.ops.aten


def _dft(self, dft_length, axis, *, inverse, onesided):
    builder = current_builder()
    if builder.opset >= 20:
        return builder.call(
            "DFT", self, dft_length, axis, inverse=inverse, onesided=onesided
        )[0]
    return builder.call(
        "DFT", self, dft_length, axis=axis, inverse=inverse, onesided=onesided
    )[0]


def _fftn_onnx_normalization(
    self: ir.Value,
    normalization: int,
    signal_size: ir.Value,
    inverse: bool = False,
) -> ir.Value:
    """Normalize in forward or backward direction."""
    builder = current_builder()
    # Norm values defined in https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOps.cpp#L117-L131
    # Norm modes: https://github.com/pytorch/pytorch/blob/758d78790164bfb041555daed380de96e06f78a3/aten/src/ATen/native/SpectralOpsUtils.h#L15-L19
    # Modes:
    # 0: no normalization (backward)
    # 1: "ortho" - divide by 1/sqrt(signal_size) (ortho)
    # 2: divide by signal_size (forward)
    signal_size = builder.call("CastLike", signal_size, self)[0]
    if not inverse:
        # Forward normalization
        if normalization == 1:
            self = builder.call("Div", self, builder.call("Sqrt", signal_size)[0])[0]
        elif normalization == 2:
            self = builder.call("Div", self, signal_size)[0]
    else:
        # Backward normalization, accounting for op.DFT already dividing by signal_size
        if normalization == 0:
            self = builder.call("Mul", self, signal_size)[0]
        elif normalization == 1:
            self = builder.call("Mul", self, builder.call("Sqrt", signal_size)[0])[0]
    return self


@onnx_impl(aten._fft_c2c.default, opset_introduced=18, complex=True)
def aten__fft_c2c(
    self: ir.Value, dim: Sequence[int], normalization: int, forward: bool
) -> ir.Value:
    """_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor

    Standard complex to complex FFT (forward or backward).
    """
    builder = current_builder()

    # NOTE: SymInt dim is not supported because DFT-17 needs a static axis

    # ONNX DFT input assumes the last dimension is the complex dimension.

    unsqueeze_first_dim = 0 in dim
    # 1. Add a new dimension for the end and batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = builder.call("Unsqueeze", self, [0])[0]
        dim = [d + 1 for d in dim]
    else:
        transformed = self

    for dimension in reversed(dim):
        transformed = _dft(
            transformed, None, dimension, inverse=not forward, onesided=False
        )
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            builder.call("Shape", transformed, start=dimension, end=dimension + 1)[0],
            not forward,
        )

    if unsqueeze_first_dim:
        transformed = builder.call("Squeeze", transformed, [0])[0]

    return transformed


@onnx_impl(aten._fft_c2r.default, opset_introduced=18, complex=True)
def aten__fft_c2r(
    self: ir.Value,
    dim: Sequence[int],
    normalization: int,
    last_dim_size: ir.Value,
) -> ir.Value:
    """_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor

    Complex to real inverse FFT. Assumes that input tensor is output of previous FFT operation.
    """
    builder = current_builder()

    dimension = dim[0]
    unsqueeze_first_dim = dimension == 0
    # 1. Add a new dimension for batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = builder.call("Unsqueeze", self, [0])[0]
    else:
        transformed = self

    for idx, dimension in enumerate(dim):
        # Adjust dimension if we unsqueezed at the beginning
        dimension += unsqueeze_first_dim

        if idx < len(dim) - 1:
            transformed = _dft(
                transformed, None, dimension, inverse=True, onesided=False
            )
        else:
            # last operation is one-sided, transform to real
            # Torch truncates/pads on the last dimension only. Typically, the only valid values that can be passed
            # into PyTorch are n or n//2+1, where n is self.shape[dim[-1]], but this is not always the case, so we
            # place no such restriction on the ONNX side.
            transformed = _dft(
                transformed, last_dim_size, dimension, inverse=True, onesided=True
            )

        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            builder.call("Shape", transformed, start=dimension, end=dimension + 1)[0],
            inverse=True,
        )

    if unsqueeze_first_dim:
        transformed = builder.call("Squeeze", transformed, [0])[0]

    # Remove the imaginary part
    transformed = builder.call("Slice", transformed, [0], [1], [-1])[0]
    transformed = builder.call("Squeeze", transformed, [-1])[0]

    return transformed


@onnx_impl(aten._fft_r2c.default, opset_introduced=18)
def aten__fft_r2c(
    self: ir.Value, dim: Sequence[int], normalization: int, onesided: bool
) -> ir.Value:
    """_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor

    Real to complex forward FFT.
    """
    builder = current_builder()

    # No need to fill the imaginary part because ONNX DFT accepts real inputs
    # https://onnx.ai/onnx/operators/onnx__DFT.html#inputs

    unsqueeze_first_dim = 0 in dim
    # 1. Add a new dimension for the end and batch dimension, if needed
    # 2. ONNX DFT input assumes the last dimension is the complex dimension.
    #       If needed, add 1 to account for the batch dimension.

    if unsqueeze_first_dim:
        transformed = builder.call("Unsqueeze", self, [0, -1])[0]
        dim = [d + 1 for d in dim]
    else:
        transformed = builder.call("Unsqueeze", self, [-1])[0]

    for idx, dimension in enumerate(reversed(dim)):
        dft_length = builder.call(
            "Squeeze",
            builder.call("Shape", transformed, start=dimension, end=dimension + 1)[0],
            [0],
        )[0]
        transformed = _fftn_onnx_normalization(
            transformed,
            normalization,
            dft_length,
            inverse=False,
        )
        if idx > 0:
            transformed = _dft(
                transformed, dft_length, dimension, inverse=False, onesided=False
            )
        else:
            # Torch computes one-sided FFT on the last dimension only.
            transformed = _dft(
                transformed, dft_length, dimension, inverse=False, onesided=onesided
            )

    if unsqueeze_first_dim:
        transformed = builder.call("Squeeze", transformed, [0])[0]

    return transformed
