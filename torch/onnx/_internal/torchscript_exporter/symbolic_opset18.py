# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    BitwiseAnd
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
    Split
"""

import functools
from collections.abc import Sequence
from typing import Optional

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = [
    "col2im",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


@_onnx_symbolic("aten::__and_")
@_onnx_symbolic("aten::bitwise_and")
def __and_(g: jit_utils.GraphContext, self, other):
    # do type promotion (scalars don't seem to apply)
    args = [self, other]
    # type promotion doesn't happen with torch.bitwise_and(tensor, scalar)
    prom_args = [arg for arg in args if symbolic_helper._get_tensor_rank(arg)]
    if len(prom_args) == 0:
        prom_args = args
    promotion_jit_type = symbolic_helper._type_promote_from_values(*prom_args)
    self = symbolic_helper._maybe_cast_to_type(g, self, promotion_jit_type)
    other = symbolic_helper._maybe_cast_to_type(g, other, promotion_jit_type)
    if promotion_jit_type == _type_utils.JitScalarType.BOOL:
        return g.op("And", self, other)
    return g.op("BitwiseAnd", self, other)


@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
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
    adjusted_padding: list[int] = []
    for pad in padding:
        adjusted_padding.extend(pad for _ in range(2))

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
    "aten::mean", decorate=[symbolic_helper._apply_params("ReduceMean", "mean")]
)
@_onnx_symbolic(
    "aten::prod",
    decorate=[
        symbolic_helper._apply_params(
            "ReduceProd", "prod", allow_multi_dim_support=False
        )
    ],
)
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
def _native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> tuple[_C.Value, _C.Value, _C.Value]:
    return opset9.native_layer_norm(g, input, normalized_shape, weight, bias, eps)


@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
def _glu(g: jit_utils.GraphContext, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", input, axis_i=dim, num_outputs_i=2, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))


@_onnx_symbolic("aten::max")
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
# TODO(justinchuby): Support multiple quantized args in output
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._max_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::maximum")
@symbolic_helper.quantized_args(True, True)
def maximum(g: jit_utils.GraphContext, input, other):
    return max(g, input, dim_or_y=other)


@_onnx_symbolic("aten::min")
# TODO(justinchuby): Support multiple quantized args in output
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._min_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::minimum")
@symbolic_helper.quantized_args(True, True)
def minimum(g: jit_utils.GraphContext, input, other):
    return min(g, input, dim_or_y=other)


@_onnx_symbolic("aten::amax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amax(g: jit_utils.GraphContext, self, dim, keepdim):
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    return g.op("ReduceMax", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::amin")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amin(g: jit_utils.GraphContext, self, dim, keepdim):
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    return g.op("ReduceMin", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::aminmax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i")
def aminmax(g: jit_utils.GraphContext, self, dim, keepdim):
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        return g.op("ReduceMin", self, axes, keepdims_i=keepdim), g.op(
            "ReduceMax", self, axes, keepdims_i=keepdim
        )
    else:
        return g.op("ReduceMin", self, keepdims_i=keepdim), g.op(
            "ReduceMax", self, keepdims_i=keepdim
        )


@_onnx_symbolic("aten::var_mean")
def _var_mean(g: jit_utils.GraphContext, input, *args):
    if len(args) == 1:
        return symbolic_helper._var_mean_helper(g, input, None, args[0], None)
    else:
        return symbolic_helper._var_mean_helper(g, input, *args)


@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
def _logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    if dim is None:
        return g.op("ReduceLogSumExp", input, keepdims_i=0)
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceLogSumExp", input, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
def _linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: list[int],
    keepdim: bool,
    dtype: torch._C.Value,
):
    return opset9.linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return symbolic_helper._embedding_bag_helper(
        g,
        embedding_matrix,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)
