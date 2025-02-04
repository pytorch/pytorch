"""torch.ops.aten operators under the `fft` module."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa
from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import INT64
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl(
    (aten._fft_c2c, aten._fft_c2r, aten._fft_r2c),
    private=True,
    complex=True,
    trace_only=True,
)
def _fftn_onnx_normalization(
    self: TFloat,
    transformed: TFloat,
    normalization: int,
    forward: bool,
    dims: Sequence[int],
) -> TFloat:
    # Obtain the total_sample_count (n) for normalization
    self_shape = op.Shape(self)
    total_sample_count = op.ReduceProd(op.Gather(self_shape, dims), keepdims=0)
    total_sample_count = op.CastLike(total_sample_count, transformed)

    # Normalize the result
    # Reference https://pytorch.org/docs/stable/generated/torch.fft.fftn.html#torch.fft.fftn
    # Reference https://github.com/pytorch/pytorch/blob/d090c18fcaaba6e1b5cb474a89058cf6081c8275/torch/_refs/fft.py#L42
    if normalization == 1:
        # "forward" - normalize by 1/n
        if forward:
            result = op.Div(transformed, op.Sqrt(total_sample_count))
        else:
            result = op.Mul(transformed, op.Sqrt(total_sample_count))
    elif normalization == 2:
        # "ortho" - normalize by 1/sqrt(n)
        if forward:
            result = op.Div(transformed, total_sample_count)
        else:
            result = transformed
    else:
        # "backward" - no normalization
        if forward:
            result = transformed
        else:
            result = op.Mul(transformed, total_sample_count)

    return result


@onnx_impl(
    (aten._fft_c2c, aten._fft_c2r, aten._fft_r2c),
    trace_only=True,
    private=True,
    complex=True,
)
def _fftn_onnx(
    self: TFloat, dims: Sequence[int], normalization: int, inverse: bool, onesided: bool
) -> TFloat:
    """Standard complex to complex or real to complex FFT (forward or backward).

    This is a private shared function for implementing the various FFT functions.

    Args:
        self: The input tensor.
        dims: The dimensions to apply FFT.
        normalization: The normalization mode.
        inverse: Whether to compute the inverse FFT.
        onesided: Whether to compute the one-sided FFT, which retains only the
            positive frequencies.

    Returns:
        The transformed tensor.
    """

    # NOTE: trace_only because we need to process each dimension in a loop
    # NOTE: SymInt dim is not support because DFT-17 needs a static axis
    # TODO(justinchuby): Make dim dynamic and remove trace_only when ONNX provides support

    # The 0-th dimension in ONNX DFT-17 is the batch dimension. We need to add a new
    # dimension at the beginning to represent the batch dimension.
    transformed = op.Unsqueeze(self, axes=[0])

    # Add 1 to account for the batch dimension when counting axes from the left
    new_dims = [dim_ + 1 if dim_ >= 0 else dim_ for dim_ in dims]

    for dim in new_dims[:-1]:
        transformed = op.DFT(transformed, axis=dim, inverse=inverse, onesided=False)

    # Torch computers one-sided FFT on the last dimension only.
    if onesided:
        transformed = op.DFT(
            transformed, axis=new_dims[-1], inverse=inverse, onesided=True
        )
    else:
        transformed = op.DFT(
            transformed, axis=new_dims[-1], inverse=inverse, onesided=False
        )

    # Remove the batch dimension
    transformed = op.Squeeze(transformed, axes=[0])

    return _fftn_onnx_normalization(self, transformed, normalization, not inverse, dims)


@onnx_impl(aten._fft_c2c, trace_only=True, complex=True)
def aten__fft_c2c(
    self: TFloat, dim: Sequence[int], normalization: int, forward: bool
) -> TFloat:
    """_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor

    Standard complex to complex FFT (forward or backward).
    """

    # NOTE: trace_only because we need to negate forward
    # NOTE: SymInt dim is not support because DFT-17 needs a static axis
    # TODO(justinchuby): Make dim dynamic and remove trace_only when ONNX provides support

    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [d - 1 if d < 0 else d for d in dim]
    return _fftn_onnx(self, dim, normalization, inverse=not forward, onesided=False)


@onnx_impl(aten._fft_c2r, trace_only=True, complex=True)
def aten__fft_c2r(
    self: TFloat,
    dim: Sequence[int],
    normalization: int,
    last_dim_size: INT64,  # pylint: disable=unused-argument
) -> TFloat:
    """_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor

    Complex to real inverse FFT.
    """

    # TODO(justinchuby): Figure out what last_dim_size does

    self_rank = len(self.shape)
    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]
    transformed = _fftn_onnx(self, dim, normalization, inverse=True, onesided=False)
    # Take only the real part
    real_part = op.Slice(transformed, axes=[-1], starts=[0], ends=[1])

    return op.Squeeze(real_part, axes=[-1])


@onnx_impl(aten._fft_r2c, trace_only=True)
def aten__fft_r2c(
    self: TFloat, dim: Sequence[int], normalization: int, onesided: bool
) -> TFloat:
    """_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor

    Real to complex forward FFT.
    """

    # Add a new dimension at the end
    signal = op.Unsqueeze(self, axes=[-1])
    # No need to fill the imaginary part because ONNX DFT accepts real inputs
    # https://onnx.ai/onnx/operators/onnx__DFT.html#inputs

    self_rank = len(self.shape)
    # ONNX DFT input assumes the last dimension is the complex dimension.
    # Thus dim=-1 in PyTorch is dim=-2 in ONNX.
    dim = [(d - 1) + self_rank if d < 0 else d for d in dim]

    return _fftn_onnx(signal, dim, normalization, inverse=False, onesided=onesided)


def aten_fft_fft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_fft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_fftfreq(n: int, d: float = 1.0) -> TensorType:
    """fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_fftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_fftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    """fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_hfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_hfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_hfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_hfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_hfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ifft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ifft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ifftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ifftshift(self: TensorType, dim: Optional[int] = None) -> TensorType:
    """fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ihfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ihfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ihfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_ihfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_ihfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_irfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_irfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_irfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_rfft(
    self: TensorType, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> TensorType:
    """fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_rfft2(
    self: TensorType,
    s: Optional[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> TensorType:
    """fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_rfftfreq(n: int, d: float = 1.0) -> TensorType:
    """fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError


def aten_fft_rfftn(
    self: TensorType,
    s: Optional[int] = None,
    dim: Optional[int] = None,
    norm: Optional[str] = None,
) -> TensorType:
    """fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"""

    raise NotImplementedError
