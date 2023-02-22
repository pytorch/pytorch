"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence

from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["col2im"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply


@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
@_beartype.beartype
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
):
    # convert [i0, i1, ..., in] into [i0, i0, i1, i1, ..., in, in]
    adjusted_padding = []
    for pad in padding:
        for _ in range(2):
            adjusted_padding.append(pad)

    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
    if not adjusted_padding:
        adjusted_padding = [0, 0] * num_dimensional_axis

    if not dilation:
        dilation = [1] * num_dimensional_axis

    if not stride:
        stride = [1] * num_dimensional_axis

    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        pads_i=adjusted_padding,
        strides_i=stride,
    )


@_onnx_symbolic(
    "aten::_upsample_bicubic2d_aa",
    decorate=[_apply_params("_upsample_bicubic2d_aa", 4, "cubic")],
)
@_onnx_symbolic(
    "aten::_upsample_bilinear2d_aa",
    decorate=[_apply_params("_upsample_bilinear2d_aa", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bicubic2d",
    decorate=[_apply_params("upsample_bicubic2d", 4, "cubic")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[_apply_params("upsample_bilinear2d", 4, "linear")],
)
@_beartype.beartype
def _interpolate(name: str, dim: int, interpolate_mode: str):
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)


@_onnx_symbolic("aten::__interpolate")
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
@symbolic_helper.parse_args("v", "v", "v", "s", "b", "none", "i")
@_beartype.beartype
def __interpolate(
    g: jit_utils.GraphContext,
    input: _C.Value,
    size: _C.Value,
    scale_factor: _C.Value,
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[_C.Value],
    antialias: Optional[int],
):
    return symbolic_helper.__interpolate_helper(
        g,
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias,
    )
