# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 9.

Opset 9 is supported by ONNX release 1.4.1
release on 01/23/19
"""

from __future__ import annotations

import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm",
    "alias",
    "amax",
    "amin",
    "aminmax",
    "arange",
    "argmax",
    "argmin",
    "as_strided",
    "as_tensor",
    "asin",
    "atan",
    "atan2",
    "baddbmm",
    "batch_norm",
    "bernoulli",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cdist",
    "ceil",
    "clamp_max",
    "clamp_min",
    "clamp",
    "clone",
    "constant_pad_nd",
    "contiguous",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "convert_element_type",
    "convolution",
    "cos",
    "cosine_similarity",
    "cross",
    "cumsum",
    "detach",
    "dim",
    "div",
    "dot",
    "dropout",
    "elu",
    "embedding_bag",
    "embedding",
    "empty_like",
    "empty",
    "eq",
    "erf",
    "exp",
    "expand_as",
    "expand",
    "eye",
    "fill",
    "flatten",
    "floor_divide",
    "floor",
    "floordiv",
    "frobenius_norm",
    "full_like",
    "full",
    "gather",
    "ge",
    "gelu",
    "get_pool_ceil_padding",
    "glu",
    "group_norm",
    "gt",
    "hann_window",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_select",
    "index",
    "instance_norm",
    "is_floating_point",
    "is_pinned",
    "isnan",
    "item",
    "kl_div",
    "layer_norm",
    "le",
    "leaky_relu",
    "lerp",
    "lift",
    "linalg_cross",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_vector_norm",
    "linear",
    "linspace",
    "log_sigmoid",
    "log_softmax",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logsumexp",
    "lstm_cell",
    "lstm",
    "lt",
    "masked_fill",
    "masked_fill_",
    "matmul",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "max",
    "maximum",
    "meshgrid",
    "min",
    "minimum",
    "mish",
    "mm",
    "movedim",
    "mse_loss",
    "mul",
    "multinomial",
    "mv",
    "narrow",
    "native_layer_norm",
    "ne",
    "neg",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "nonzero_numpy",
    "nonzero",
    "norm",
    "numel",
    "numpy_T",
    "one_hot",
    "ones_like",
    "ones",
    "onnx_placeholder",
    "pad",
    "pairwise_distance",
    "permute",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pow",
    "prelu",
    "prim_constant_chunk",
    "prim_constant_split",
    "prim_constant",
    "prim_data",
    "prim_device",
    "prim_dtype",
    "prim_if",
    "prim_layout",
    "prim_list_construct",
    "prim_list_unpack",
    "prim_loop",
    "prim_max",
    "prim_min",
    "prim_shape",
    "prim_tolist",
    "prim_tuple_construct",
    "prim_type",
    "prim_unchecked_cast",
    "prim_uninitialized",
    "rand_like",
    "rand",
    "randint_like",
    "randint",
    "randn_like",
    "randn",
    "reciprocal",
    "reflection_pad",
    "relu",
    "relu6",
    "remainder",
    "repeat_interleave",
    "repeat",
    "replication_pad",
    "reshape_as",
    "reshape",
    "roll",
    "rrelu",
    "rsqrt",
    "rsub",
    "scalar_tensor",
    "scatter_add",
    "scatter",
    "select",
    "selu",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "size",
    "slice",
    "softmax",
    "softplus",
    "softshrink",
    "sort",
    "split_with_sizes",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std_mean",
    "std",
    "sub",
    "t",
    "take",
    "tan",
    "tanh",
    "tanhshrink",
    "tensor",
    "threshold",
    "to",
    "topk",
    "transpose",
    "true_divide",
    "type_as",
    "unbind",
    "unfold",
    "unsafe_chunk",
    "unsafe_split_with_sizes",
    "unsafe_split",
    "unsqueeze",
    "unsupported_complex_operators",
    "noop_complex_operators",
    "unused",
    "var_mean",
    "var",
    "view_as",
    "view",
    "where",
    "wrap_logical_op_with_cast_to",
    "wrap_logical_op_with_negation",
    "zeros_like",
    "zeros",
    "zero",
]


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)


def _export(name: str):
    """Exports the function in the current global namespace."""

    def wrapper(func):
        globals()[name] = func
        __all__.append(name)
        return func

    return wrapper


@_beartype.beartype
def unused(g):
    """Represents "missing" optional inputs."""
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


@_onnx_symbolic("aten::_shape_as_tensor")
@_beartype.beartype
def _shape_as_tensor(g: jit_utils.GraphContext, input):
    return g.op("Shape", input)


@_onnx_symbolic("aten::_reshape_from_tensor")
@_beartype.beartype
def _reshape_from_tensor(g: jit_utils.GraphContext, input, shape):
    if isinstance(shape, list):
        shape = g.op("Concat", *shape, axis_i=0)
    return reshape(g, input, shape)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def reshape(g: jit_utils.GraphContext, self, shape):
    return symbolic_helper._reshape_helper(g, self, shape)


@_onnx_symbolic("aten::reshape_as")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def reshape_as(g: jit_utils.GraphContext, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@_onnx_symbolic("aten::add")
@_beartype.beartype
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    """
    This function takes the add function and returns the corresponding ONNX operator.

    This function is not meant to be called directly by the user.

    Args:
        g (GraphContext): The graph context.
        self (Tensor): The first operand.
        other (Tensor): The second operand.
        alpha (float, optional): The scaling factor for the second operand. Defaults to None.

    Returns:
        ONNX operator.
    """
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Add", 9, 11, "Add between list of tensors not supported", self
        )
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Add", self, other)


@_onnx_symbolic("aten::sub")
@_beartype.beartype
def sub(g: jit_utils.GraphContext, self, other, alpha=None):
    """
    Consumes sub function and returns the corresponding ONNX operator.

    This function is not meant to be called directly by the user.

    Args:
        g (GraphContext): The graph context.
        self (Tensor): The first operand.
        other (Tensor): The second operand.
        alpha (Optional[Tensor]): A scaling factor to apply to the second operand.
            If `alpha` is not provided, it defaults to 1.

    Returns:
        ONNX operator
    """
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Sub", self, other)


@_onnx_symbolic("aten::rsub")
@_beartype.beartype
def rsub(g: jit_utils.GraphContext, self, other, alpha=None):
    return sub(g, other, self, alpha=alpha)


@_onnx_symbolic("aten::mul")
@_beartype.beartype
def mul(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_bool(self) and symbolic_helper._is_bool(other):
        # ONNX Mul doesn't support Boolean, so use And as an equivalent operator.
        return g.op("And", self, other)
    else:
        return g.op("Mul", self, other)


@_onnx_symbolic("aten::div")
@_beartype.beartype
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        return true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@_onnx_symbolic("aten::addcmul")
@symbolic_helper.parse_args("v", "v", "v", "f")
@_beartype.beartype
def addcmul(g: jit_utils.GraphContext, self, tensor1, tensor2, value=1.0):
    value_tens = g.op("Constant", value_t=torch.tensor([value]))
    return add(g, self, mul(g, mul(g, tensor1, tensor2), value_tens))


@symbolic_helper.parse_args("v", "v", "s")
@_beartype.beartype
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode is None:
        return true_divide(g, self, other)
    elif rounding_mode == "floor":
        return _floor_divide(g, self, other)
    elif rounding_mode == "trunc":
        return _trunc_divide(g, self, other)
    else:
        raise errors.SymbolicValueError(
            f'Unsupported rounding mode: "{rounding_mode}". Expected None, "floor" or "trunc"',
            self,
        )


@_beartype.beartype
def _trunc_divide(g: jit_utils.GraphContext, self, other):
    out = g.op("Div", self, other)
    # the correct operation is truncate, which is not supported in ONNX,
    # we cannot call floor since it will behave differently for negative numbers
    # (eg. -0.1 should become -0 )
    # - if scalar_type information are not available, assume that
    # we need to call floor (treat as float)
    out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.INT64)

    # Matching PyTorch's behavior:
    # - if self is fp the output's type is self's type
    # - if self is not fp and other is fp, the output is of type JitScalarType.FLOAT
    # - self is not fp and other is not fp, the output's type is self's output type
    # - the output type defaults to Float
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        if not symbolic_helper._is_fp(self) and symbolic_helper._is_fp(other):
            out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        else:
            out = g.op(
                "Cast",
                out,
                to_i=scalar_type.onnx_type(),
            )
    else:
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return out


@_beartype.beartype
def _floor_divide(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does trunction rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op(
            "Xor",
            symbolic_helper._lt_helper(g, self, zero),
            symbolic_helper._lt_helper(g, other, zero),
        )

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Sub", self, g.op("Mul", div, other))
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Mul", fixup_mask, one)
        return g.op("Sub", div, fixup)


@_onnx_symbolic("aten::floor_divide")
@_beartype.beartype
def floor_divide(g: jit_utils.GraphContext, self, other):
    # Deprecated behavior, floor_divide actually truncates
    return _trunc_divide(g, self, other)


@_onnx_symbolic("aten::floordiv")
@_beartype.beartype
def floordiv(g: jit_utils.GraphContext, self, other):
    return floor_divide(g, self, other)


@_onnx_symbolic("aten::true_divide")
@_beartype.beartype
def true_divide(g: jit_utils.GraphContext, self, other):
    """Division where both inputs are cast to floating types

    If both inputs are floating, performs div as usual
    If only one input is a floating type, the other input is cast to its type
    If neither input is a floating type, both inputs are cast to the default scalar type
    """

    # Case 1: either values are floating
    # Performs div as usual.
    # Implicit casting will be handled in scalar type analysis pass.
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        return g.op("Div", self, other)

    # Case 2: neither is floating
    # Casts both inputs to the default scalar type
    scalar_type = torch.get_default_dtype()
    onnx_scalar_type = _C_onnx.TensorProtoDataType.FLOAT
    assert scalar_type is torch.float or scalar_type is torch.double
    if torch.get_default_dtype() is torch.double:
        onnx_scalar_type = _C_onnx.TensorProtoDataType.DOUBLE

    self = g.op("Cast", self, to_i=onnx_scalar_type)
    other = g.op("Cast", other, to_i=onnx_scalar_type)
    return g.op("Div", self, other)


@_onnx_symbolic("aten::reciprocal")
@_beartype.beartype
def reciprocal(g: jit_utils.GraphContext, self):
    # torch.reciprocal implicitly casts to float, so we do the same.
    if not symbolic_helper._is_fp(self):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return g.op("Reciprocal", self)


@_onnx_symbolic("aten::cat")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    """Implement concatenation of pytorch tensors in ONNX along the specified `dim` dimension.

    Parameters:
        g (jit_utils.GraphContext): Graph context.
        tensor_list (List[torch.Tensor]): List of tensors to concatenate.
        dim (int): Dimension along which to concatenate the tensors.

    Returns:
        ONNX graph node representing the concatenated tensor.
    """
    tensors = symbolic_helper._unpack_list(tensor_list)
    # torch.cat ignores empty tensors such as `torch.Tensor([])`
    # These needs to be removed as input from ONNX's concat too, otherwise shape inference
    # will likely fail due to inputs with different ranks (0 for empty tensor, > 0 for anything else)
    nonempty_tensors = []
    for t in tensors:
        if symbolic_helper._is_constant(t) and not symbolic_helper._get_tensor_dim_size(
            t, 0
        ):
            continue
        nonempty_tensors.append(t)
    assert len(nonempty_tensors) > 0
    assert all(
        symbolic_helper._get_tensor_rank(nonempty_tensors[0]) is None
        or symbolic_helper._get_tensor_rank(t) is None
        or symbolic_helper._get_tensor_rank(t)
        == symbolic_helper._get_tensor_rank(nonempty_tensors[0])
        for t in nonempty_tensors
    )
    tensor_list.node().removeAllInputs()
    for t in nonempty_tensors:
        tensor_list.node().addInput(t)

    tensors = symbolic_helper._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@_onnx_symbolic("aten::stack")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    unsqueezed = [
        symbolic_helper._unsqueeze_helper(g, t, [dim])
        for t in symbolic_helper._unpack_list(tensor_list)
    ]
    return g.op("Concat", *unsqueezed, axis_i=dim)


@_onnx_symbolic("aten::list")
@_beartype.beartype
def _list(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("aten::mm")
@_beartype.beartype
def mm(g: jit_utils.GraphContext, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    C = g.op("Constant", value_t=torch.tensor([1]))
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::bmm")
@_beartype.beartype
def bmm(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::matmul")
@_beartype.beartype
def matmul(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
@_beartype.beartype
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha):
    scalar_type = None
    self_scalar_type = symbolic_helper._try_get_scalar_type(self)
    mat1_scalar_type = symbolic_helper._try_get_scalar_type(mat1)
    mat2_scalar_type = symbolic_helper._try_get_scalar_type(mat2)
    if self_scalar_type is not None:
        scalar_type = self_scalar_type
    elif mat1_scalar_type is not None:
        scalar_type = mat1_scalar_type
    elif mat2_scalar_type is not None:
        scalar_type = mat2_scalar_type

    mat1_rank = symbolic_helper._get_tensor_rank(mat1)
    mat2_rank = symbolic_helper._get_tensor_rank(mat2)

    def is_not_none_nor(v, u):
        return v is not None and v != u

    if scalar_type is not None and (
        is_not_none_nor(mat1_rank, 2) or is_not_none_nor(mat2_rank, 2)
    ):
        res1 = g.op("MatMul", mat1, mat2)
        res2 = self

        alpha = symbolic_helper._scalar(alpha)
        beta = symbolic_helper._scalar(beta)

        if alpha != 1:
            alpha = g.op(
                "Constant", value_t=torch.tensor(alpha, dtype=scalar_type.dtype())
            )
            res1 = g.op("Mul", res1, alpha)
        if beta != 1:
            beta = g.op(
                "Constant",
                value_t=torch.tensor(
                    symbolic_helper._scalar(beta), dtype=scalar_type.dtype()
                ),
            )
            res2 = g.op("Mul", res2, beta)

        return g.op("Add", res1, res2)

    return g.op(
        "Gemm",
        mat1,
        mat2,
        self,
        beta_f=symbolic_helper._scalar(beta),
        alpha_f=symbolic_helper._scalar(alpha),
    )


@_onnx_symbolic("aten::neg")
@_beartype.beartype
def neg(g: jit_utils.GraphContext, self):
    return g.op("Neg", self)


@_onnx_symbolic("aten::sqrt")
@_beartype.beartype
def sqrt(g: jit_utils.GraphContext, self):
    if _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
        _type_utils.JitScalarType.INT16,
        _type_utils.JitScalarType.INT,
        _type_utils.JitScalarType.INT64,
    }:
        # torch converts all int inputs to sqrt to float
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    return g.op("Sqrt", self)


@_onnx_symbolic("aten::rsqrt")
@_beartype.beartype
def rsqrt(g: jit_utils.GraphContext, self):
    return g.op(
        "Div", symbolic_helper._if_scalar_type_as(torch.ones(1), self), sqrt(g, self)
    )


@_onnx_symbolic("aten::tanh")
# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qtanh.cpp
@symbolic_helper.quantized_args(True, scale=2.0 / 256.0, zero_point=128)
@_beartype.beartype
def tanh(g: jit_utils.GraphContext, self):
    return g.op("Tanh", self)


@_onnx_symbolic("aten::sin")
@_beartype.beartype
def sin(g: jit_utils.GraphContext, self):
    return g.op("Sin", self)


@_onnx_symbolic("aten::cos")
@_beartype.beartype
def cos(g: jit_utils.GraphContext, self):
    return g.op("Cos", self)


@_onnx_symbolic("aten::tan")
@_beartype.beartype
def tan(g: jit_utils.GraphContext, self):
    return g.op("Tan", self)


@_onnx_symbolic("aten::asin")
@_beartype.beartype
def asin(g: jit_utils.GraphContext, self):
    return g.op("Asin", self)


@_onnx_symbolic("aten::acos")
@_beartype.beartype
def acos(g: jit_utils.GraphContext, self):
    return g.op("Acos", self)


@_onnx_symbolic("aten::atan")
@_beartype.beartype
def atan(g: jit_utils.GraphContext, self):
    return g.op("Atan", self)


@_onnx_symbolic("aten::atan2")
@_beartype.beartype
def atan2(g: jit_utils.GraphContext, self, other):
    # self is y, and other is x on coordinate
    slope = g.op("Div", self, other)
    atan = g.op("Atan", slope)
    const_zero = g.op("Constant", value_t=torch.tensor(0))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi))

    condition_second_or_third_quadrant = g.op("Greater", self, const_zero)
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result


@_onnx_symbolic("aten::sigmoid")
# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qsigmoid.cpp
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
@_beartype.beartype
def sigmoid(g: jit_utils.GraphContext, self):
    """Converts the corresponding PyTorch function into ONNX operators.

    It is not meant to be called directly by a user.

    Args:
        g (jit_utils.GraphContext): Graph context.
        self (Tensor): the input tensor.
    Returns:
        ONNX operator
    """
    return g.op("Sigmoid", self)


@_onnx_symbolic("aten::sign")
@_beartype.beartype
def sign(g: jit_utils.GraphContext, self):
    return g.op("Sign", self)


@symbolic_helper.quantized_args(True)
@_beartype.beartype
def _slice(g: jit_utils.GraphContext, input, axes, starts, ends):
    assert len(starts) == len(ends)
    if len(starts) == 1 and starts[0] == 0 and ends[0] == _constants.INT64_MAX:
        return input
    return g.op("Slice", input, axes_i=axes, starts_i=starts, ends_i=ends)


@_onnx_symbolic(
    "aten::sum", decorate=[symbolic_helper._apply_params("ReduceSum", "sum")]
)
@_onnx_symbolic(
    "aten::mean", decorate=[symbolic_helper._apply_params("ReduceMean", "mean")]
)
# torch.prod does not support multidimensional "dim"
@_onnx_symbolic(
    "aten::prod",
    decorate=[
        symbolic_helper._apply_params(
            "ReduceProd", "prod", allow_multi_dim_support=False
        )
    ],
)
@_beartype.beartype
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def cumsum(g: jit_utils.GraphContext, input, dim, dtype):
    if symbolic_helper.is_caffe2_aten_fallback():
        if dtype.node().kind() != "prim::Constant":
            return symbolic_helper._unimplemented("cumsum", "dtype", dtype)
        return g.at("cumsum", input, dim_i=dim)

    symbolic_helper._onnx_opset_unsupported("cumsum", 9, 11, input)


@_onnx_symbolic("aten::_sample_dirichlet")
@_beartype.beartype
def _sample_dirichlet(g: jit_utils.GraphContext, self, generator):
    if symbolic_helper.is_caffe2_aten_fallback():
        if not symbolic_helper._is_none(generator):
            return symbolic_helper._unimplemented(
                "_sample_dirichlet", "We are not able to export generator", self
            )
        return g.at("_sample_dirichlet", self)
    return symbolic_helper._onnx_unsupported("_sample_dirichlet", self)


@_onnx_symbolic("aten::_standard_gamma")
@_beartype.beartype
def _standard_gamma(g: jit_utils.GraphContext, self, generator):
    if symbolic_helper.is_caffe2_aten_fallback():
        if not symbolic_helper._is_none(generator):
            return symbolic_helper._unimplemented(
                "_standard_gamma", "not able to export generator", self
            )
        return g.at("_standard_gamma", self)

    return symbolic_helper._onnx_unsupported("_standard_gamma", self)


@_onnx_symbolic("aten::t")
@_beartype.beartype
def t(g: jit_utils.GraphContext, self):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is None or rank < 2:
        # The transpose of a 1d or 0d tensor is itself. ONNX does not define the behavior
        # clearly and onnxruntime fails on these cases. So we add an Identity node to
        # mirror the behavior of eager mode.
        return g.op("Identity", self)
    return g.op("Transpose", self, perm_i=(1, 0))


@_onnx_symbolic("aten::numpy_T")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def numpy_T(g: jit_utils.GraphContext, input):
    ndim = symbolic_helper._get_tensor_rank(input)
    assert ndim is not None
    perm = list(reversed(range(0, ndim)))
    return g.op("Transpose", input, perm_i=perm)


@_onnx_symbolic("aten::expand")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def expand(g: jit_utils.GraphContext, self, size, implicit):
    """Implement the expand function for a pytorch tensor in ONNX according to specified `size`"""
    size = symbolic_helper._maybe_get_const(size, "is")
    if not symbolic_helper._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif symbolic_helper._is_packed_list(size):
        # Expand with -1 dim value means dim is unchanged.
        # Since onnx::expand supports two-way broadcasting,
        # -1 dim value can be exported to onnx as 1
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    dtype = _type_utils.JitScalarType.INT64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


@_onnx_symbolic("aten::broadcast_to")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def broadcast_to(g: jit_utils.GraphContext, self, size):
    size = symbolic_helper._maybe_get_const(size, "is")
    if not symbolic_helper._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif symbolic_helper._is_packed_list(size):
        # Expand with -1 dim value means dim is unchanged.
        # Since onnx::expand supports two-way broadcasting,
        # -1 dim value can be exported to onnx as 1
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    dtype = _type_utils.JitScalarType.INT64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


@_onnx_symbolic("aten::expand_as")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def expand_as(g: jit_utils.GraphContext, self, other):
    self_t = symbolic_helper._maybe_get_const(self, "t")
    if isinstance(self_t, torch.Tensor):
        orig_type = self_t.dtype
        self_t = self_t.to(torch.double)
        dims = []
        for d in range(self_t.dim()):
            if torch.equal(self_t.mean(d).unsqueeze(d).expand_as(self_t), self_t):
                dims.append(d)
                self = g.op(
                    "Constant", value_t=self_t.mean(dims, keepdim=True).to(orig_type)
                )

    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


@_onnx_symbolic("aten::embedding")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i", "b", "v")
@_beartype.beartype
def embedding(
    g: jit_utils.GraphContext,
    weight,
    indices,
    padding_idx,
    scale_grad_by_freq,
    sparse,
):
    if scale_grad_by_freq and GLOBALS.export_training:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of embedding with scale_grad_by_freq=True "
            "for training mode. ONNX does not support scaling the gradients.",
            weight,
        )
    if padding_idx >= 0 and GLOBALS.export_training:
        warnings.warn(
            "Warning: ONNX export of embedding with padding_idx >= 0 "
            "for training mode. "
            "ONNX does not support not updating the embedding vector at padding_idx during training."
        )

    return g.op("Gather", weight, indices)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
@_beartype.beartype
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
    if not symbolic_helper._is_none(per_sample_weights):
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with per_sample_weights"
        )
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "embedding_bag",
            embedding_matrix,
            indices,
            offsets,
            outputs=4,
            scale_grad_by_freq_i=scale_grad_by_freq,
            mode_i=mode,
            sparse_i=sparse,
            include_last_offset_i=include_last_offset,
            padding_idx_i=padding_idx,
        )

    return symbolic_helper._onnx_unsupported("embedding_bag", embedding_matrix)


@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
@_beartype.beartype
def size(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    if symbolic_helper._maybe_get_const(dim, "i") < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            dim = symbolic_helper._maybe_get_const(dim, "i") + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    return symbolic_helper._size_helper(g, self, dim)


@_onnx_symbolic("aten::transpose")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def transpose(g: jit_utils.GraphContext, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return g.op("Transpose", self, perm_i=axes)
    elif symbolic_helper.is_caffe2_aten_fallback():
        # if we don't have dim information we cannot
        # output a permute so use ATen instead
        return g.at("transpose", self, overload_name="int", dim0_i=dim0, dim1_i=dim1)
    else:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of transpose for tensor of unknown rank.",
            self,
        )


@_onnx_symbolic("aten::permute")
@symbolic_helper.parse_args("v", "is")
@_beartype.beartype
def permute(g: jit_utils.GraphContext, self, dims):
    if dims == list(range(0, len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


@_onnx_symbolic("aten::view")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def view(g: jit_utils.GraphContext, self, size):
    return reshape(g, self, size)


@_onnx_symbolic("aten::view_as")
@_beartype.beartype
def view_as(g: jit_utils.GraphContext, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unsafe_chunk", 9, 11, "Dynamic number of outputs not supported", self
        )
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented(
            "unsafe_chunk", "unknown dimension size", self
        )
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split", 9, 11, "Dynamic number of outputs not supported", self
        )
    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    if split_val.dim() > 0:
        return split_with_sizes(g, self, split_size_or_sizes, dim, _outputs)
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "split", 9, 11, "Unknown dimension size not supported", self
            )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unsafe_split")
@_beartype.beartype
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "is", "i", "i")
@_beartype.beartype
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split_with_sizes", 9, 11, "Dynamic number of outputs not supported", self
        )
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unsafe_split_with_sizes")
@_beartype.beartype
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unbind", 9, 11, "Dynamic number of outputs not supported", self
        )

    outputs = g.op("Split", self, split_i=[1] * _outputs, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [
        symbolic_helper._squeeze_helper(g, out, [dim]) for out in outputs
    ]
    return squeezed_outputs


@_onnx_symbolic("aten::select")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
@_beartype.beartype
def select(g: jit_utils.GraphContext, self, dim, index):
    """Implement the select functionality for a pytorch tensor in ONNX.

    Selects elements from the input tensor along the specified `dim` dimension based on the `index` tensor.
    """
    index = symbolic_helper._maybe_get_scalar(index)
    if (not symbolic_helper._is_value(index)) and (index < 0):
        if index == -1:
            end_index = _constants.INT64_MAX
        else:
            end_index = index + 1
        slice_node = symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[index], ends=[end_index]
        )
        return symbolic_helper._squeeze_helper(g, slice_node, [dim])
    else:
        # FIXME(justinchuby): can index be an int and not a value?
        return g.op("Gather", self, index, axis_i=dim)


@_onnx_symbolic("aten::square")
@_beartype.beartype
def square(g: jit_utils.GraphContext, self):
    return g.op("Mul", self, self)


@_onnx_symbolic("aten::squeeze")
@_beartype.beartype
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    squeeze_dim = symbolic_helper._get_const(dim, "i", "dim")
    # Handle negative dims
    if squeeze_dim < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            warnings.warn(
                "ONNX export squeeze with negative axis "
                + str(squeeze_dim)
                + " might cause the onnx model to be incorrect. "
                + "Negative axis is not supported in ONNX. "
                + "Axis is converted to "
                + str(squeeze_dim + rank)
                + " based on input shape at export time. "
                + "Passing an tensor of different rank in execution will be incorrect."
            )
            squeeze_dim += rank
        else:
            return symbolic_helper._unimplemented(
                "squeeze", "negative axis with unknown input rank", self
            )

    dim_size = symbolic_helper._get_tensor_dim_size(self, squeeze_dim)
    if dim_size is None:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + " on an input "
            + "with unknown shape. Note that if the size of dimension "
            + str(squeeze_dim)
            + " of the input "
            + "is not 1, the ONNX model will return an error. Opset version 11 supports squeezing on "
            + "non-singleton dimensions, it is recommended to export this model using opset "
            + "version 11 or higher."
        )
        return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])
    if dim_size > 1:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please use opset version 11 to "
            + "export the model."
        )
        return self

    warnings.warn(
        "This model contains a squeeze operation on dimension "
        + str(squeeze_dim)
        + ". If the model is "
        + "intended to be used with dynamic input shapes, please use opset version 11 to export the model."
    )
    return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])


@_onnx_symbolic("aten::prelu")
@_beartype.beartype
def prelu(g: jit_utils.GraphContext, self, weight):
    self_rank = symbolic_helper._get_tensor_rank(self)
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    weight_rank = len(weight_sizes)
    if self_rank is not None:
        if self_rank > 2:
            # make weight unidirectional broadcastable
            weight = symbolic_helper._unsqueeze_helper(
                g, weight, list(range(1, self_rank - 1))
            )
        elif self_rank == 0 and weight_sizes == [1]:
            # self and weight are both scalar but weight has rank == 1, squeeze weight.
            weight = symbolic_helper._squeeze_helper(g, weight, [0])
            weight_rank = 0

    if self_rank is not None and weight_rank is not None:
        assert (
            self_rank >= weight_rank
        ), f"rank(x) should be >= rank(slope) but got {self_rank} < {weight_rank}"
    return g.op("PRelu", self, weight)


@_onnx_symbolic("aten::silu")
@_beartype.beartype
def silu(g: jit_utils.GraphContext, input):
    return g.op("Mul", input, g.op("Sigmoid", input))


@_onnx_symbolic("aten::mish")
@_beartype.beartype
def mish(g: jit_utils.GraphContext, input):
    return g.op("Mul", input, g.op("Tanh", g.op("Softplus", input)))


@_onnx_symbolic("aten::relu")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def relu(g: jit_utils.GraphContext, input):
    return symbolic_helper._op_with_optional_float_cast(
        g, "Relu", input, opset_before=14
    )


@_onnx_symbolic("aten::relu6")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def relu6(g: jit_utils.GraphContext, input):
    return clamp(g, input, 0, 6)


@_onnx_symbolic("aten::ceil")
@_beartype.beartype
def ceil(g: jit_utils.GraphContext, input):
    return g.op("Ceil", input)


@_onnx_symbolic("aten::floor")
@_beartype.beartype
def floor(g: jit_utils.GraphContext, input):
    return g.op("Floor", input)


@_onnx_symbolic("aten::len")
@_beartype.beartype
def _len(g: jit_utils.GraphContext, self):
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@_onnx_symbolic("aten::threshold")
@symbolic_helper.parse_args("v", "t", "t")
@_beartype.beartype
def threshold(g: jit_utils.GraphContext, self, threshold, value):
    # See Note [Export inplace]
    if symbolic_helper._scalar(threshold) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero threshold", self)
    if symbolic_helper._scalar(value) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero value", self)
    return g.op("Relu", self)


@_onnx_symbolic("aten::leaky_relu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "b")
@_beartype.beartype
def leaky_relu(
    g: jit_utils.GraphContext,
    input: _C.Value,
    negative_slope: float,
    inplace: bool = False,
):
    # See Note [Export inplace]
    return g.op("LeakyRelu", input, alpha_f=negative_slope)


@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def glu(g: jit_utils.GraphContext, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", input, axis_i=dim, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))


@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # Softmax does normalization at vector level.
    # PyTorch and ONNX use different strategies to split the input tensor into vectors.
    # Thus dim and axis have different meanings.
    # PyTorch slices the input tensor into vectors along the `dim`-th dimension.
    # ONNX reshapes the input into a 2-D tensor, and `axis` indicates where the input is coerced.
    # If input is a 2 x 3 tensor:
    # input = [[1.0, 1.0, 1.0],
    #          [1.0, 1,0, 1,0]]
    # with dim = 0, the result is:
    # result = [[0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5]]
    # with axis = 0, the result is:
    # result = [[0.167, 0.167, 0.167],
    #           [0.167, 0.167, 0.167]]
    # So only when dim and axis both equal to ndim - 1 (the last dimension),
    # their semantics are equivalent.
    # So use softmax when dim and axis both equal to ndim - 1,
    # otherwise transpose the input to put the vectors to be normalized to the last dimension.
    # When input rank is not known at export time we compute softmax using a subgraph
    # with other operators
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is not None:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        if dim < 0:
            dim = input_dim + dim

        is_transpose_required = input_dim != dim + 1

        if is_transpose_required:
            axes = list(range(input_dim))
            axes[dim], axes[-1] = axes[-1], axes[dim]
            input = g.op("Transpose", input, perm_i=axes)
            dim = input_dim - 1

        softmax = g.op("Softmax", input, axis_i=dim)
        if dtype and dtype.node().kind() != "prim::Constant":
            parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
            softmax = g.op(
                "Cast",
                softmax,
                to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type(),
            )

        if is_transpose_required:
            softmax = g.op("Transpose", softmax, perm_i=axes)  # type: ignore[possibly-undefined]
        return softmax

    # Apply max normalization.
    input = g.op("Sub", input, g.op("ReduceMax", input, axes_i=[dim], keepdims_i=1))

    exp = g.op("Exp", input)
    sum = symbolic_helper._reducesum_helper(g, exp, axes_i=[dim])
    softmax = g.op("Div", exp, sum)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    return softmax


@_onnx_symbolic("aten::softplus")
@_beartype.beartype
def softplus(g: jit_utils.GraphContext, self, beta, threshold):
    beta_const = symbolic_helper._maybe_get_const(beta, "f")
    if beta_const != 1:
        return g.op("Div", g.op("Softplus", g.op("Mul", self, beta)), beta)
    return g.op("Softplus", self)


@_onnx_symbolic("aten::get_pool_ceil_padding")
@_beartype.beartype
def get_pool_ceil_padding(input, kernel_size, stride, padding):
    # TODO(justinchuby): Looks like this op is deprecated in torch
    sizes = symbolic_helper._get_tensor_sizes(input)
    dim = sizes[-len(padding) :] if sizes is not None else None
    if dim is None or any(i is None for i in dim):
        return symbolic_helper._unimplemented(
            "get_pool_ceil_padding", "input size not accessible", input
        )
    ceiled_output_dim = [
        int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i])))
        + 1
        for i in range(0, len(padding))
    ]
    # ensure last pooling starts inside
    ceiled_output_dim = [
        (
            ceiled_output_dim[i] - 1
            if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
            else ceiled_output_dim[i]
        )
        for i in range(0, len(ceiled_output_dim))
    ]
    padding_ceil = [
        (
            0
            if (stride[i] == 1)
            else (
                kernel_size[i]
                - (
                    dim[i]
                    + 2 * padding[i]
                    - ((ceiled_output_dim[i] - 1) * stride[i] + 1)
                )
            )
        )
        for i in range(0, len(padding))
    ]
    # ensure padding is not > kernel_size
    padding_ceil = [
        (
            (
                int(padding_ceil[i])
                if padding_ceil[i] < kernel_size[i] - 1
                else int(kernel_size[i] - 1)
            )
            if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
            else int(padding_ceil[i])
        )
        for i in range(0, len(padding_ceil))
    ]
    return padding_ceil


@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d", torch.nn.modules.utils._single, 1, return_indices=False
        ),
        _export("max_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d", torch.nn.modules.utils._pair, 2, return_indices=False
        ),
        _export("max_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d", torch.nn.modules.utils._triple, 3, return_indices=False
        ),
        _export("max_pool3d"),
    ],
)
@_beartype.beartype
def _max_pool(name, tuple_fn, ndims, return_indices):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    @_beartype.beartype
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if set(tuple_fn(dilation)) != {1}:
            return symbolic_helper._unimplemented(name, "dilation", input)
        if not stride:
            stride = kernel_size
        padding = tuple(tuple_fn(padding))
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            padding = padding + tuple(a + b for (a, b) in zip(padding_ceil, padding))
        else:
            padding = padding * 2
        kwargs = {
            "kernel_shape_i": tuple_fn(kernel_size),
            "pads_i": padding,
            "strides_i": tuple_fn(stride),
        }
        # easy but hacky way to get flattened indices values
        # to be used to convert the indices values to non-flattened.
        # In ONNX the indices are computed as a flatten 1-D tensor,
        # so the values in indices are in [0, N x C x D1 x ... x Dn).
        # To convert the indices to the same format used by Pytorch,
        # we first execute a maxpool with a kernel and stride of 1 on the same input.
        # This will result in a tensor of indices in which each index will have it's own value.
        # Using this tensor as a reference, we extract the first index of each axis and subtract
        # it from each index of this axis in the indices to convert.
        # This step will result in a tensor were each dimension has values of indices within
        # the dimension it is in.
        # For more information :
        # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
        if return_indices:
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            _, flattened_indices = g.op(
                "MaxPool",
                input,
                outputs=2,
                kernel_shape_i=[1 for _ in range(ndims)],
                strides_i=[1 for _ in range(ndims)],
            )
            # convert indices to have non-flattened indices values
            s = symbolic_helper._slice_helper(
                g,
                flattened_indices,
                axes=[2 + i for i in range(ndims)],
                starts=list(tuple_fn(0)),
                ends=list(tuple_fn(1)),
            )
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d_with_indices = _onnx_symbolic("aten::max_pool1d_with_indices")(
    _max_pool(
        "max_pool1d_with_indices",
        torch.nn.modules.utils._single,
        1,
        return_indices=True,
    )
)
max_pool2d_with_indices = _onnx_symbolic("aten::max_pool2d_with_indices")(
    _max_pool(
        "max_pool2d_with_indices",
        torch.nn.modules.utils._pair,
        2,
        return_indices=True,
    )
)
max_pool3d_with_indices = _onnx_symbolic("aten::max_pool3d_with_indices")(
    _max_pool(
        "max_pool3d_with_indices",
        torch.nn.modules.utils._triple,
        3,
        return_indices=True,
    )
)


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[
        symbolic_helper._apply_params("avg_pool1d", torch.nn.modules.utils._single),
        _export("avg_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[
        symbolic_helper._apply_params("avg_pool2d", torch.nn.modules.utils._pair),
        _export("avg_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[
        symbolic_helper._apply_params("avg_pool3d", torch.nn.modules.utils._triple),
        _export("avg_pool3d"),
    ],
)
@_beartype.beartype
def _avg_pool(name, tuple_fn):
    @symbolic_helper.quantized_args(True)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    @_beartype.beartype
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        assert isinstance(padding, tuple)
        adjusted_padding = padding
        # Although onnx::AvgPool provides count_include_pad,
        # The corner case of Average Pooling with ceil_mode on
        # PyTorch allows sliding window go off bound, which leads to
        # this accommodation.
        # More detail on https://github.com/pytorch/pytorch/issues/57178
        if count_include_pad:
            input = symbolic_helper._op_with_optional_float_cast(
                g,
                "Pad",
                input,
                pads_i=((0,) * 2 + padding) * 2,
                mode_s="constant",
                value_f=0.0,
                opset_before=11,
            )
            adjusted_padding = (0,) * len(padding)
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            adjusted_padding = adjusted_padding + tuple(
                a + b for (a, b) in zip(padding_ceil, adjusted_padding)
            )
        else:
            adjusted_padding = adjusted_padding * 2
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=adjusted_padding,
        )
        return output

    return symbolic_fn


@_onnx_symbolic(
    "aten::adaptive_avg_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool1d", "AveragePool", torch.nn.modules.utils._single
        ),
        _export("adaptive_avg_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_avg_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool2d", "AveragePool", torch.nn.modules.utils._pair
        ),
        _export("adaptive_avg_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_avg_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool3d", "AveragePool", torch.nn.modules.utils._triple
        ),
        _export("adaptive_avg_pool3d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool1d",
            "MaxPool",
            torch.nn.modules.utils._single,
            max_pool1d_with_indices,
        ),
        _export("adaptive_max_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool2d",
            "MaxPool",
            torch.nn.modules.utils._pair,
            max_pool2d_with_indices,
        ),
        _export("adaptive_max_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool3d",
            "MaxPool",
            torch.nn.modules.utils._triple,
            max_pool3d_with_indices,
        ),
        _export("adaptive_max_pool3d"),
    ],
)
@_beartype.beartype
def _adaptive_pool(name, type, tuple_fn, fn=None):
    @symbolic_helper.quantized_args(True, False)
    @_beartype.beartype
    def symbolic_fn(g, input, output_size):
        # _adaptive_pool is supported for cases where output_size is 1 for all dimensions,
        # by executing a GlobalPool.
        # It is also supported for cases where the output size is a factor of the input size.
        # For these cases the stride and kernel size are uniform along all the indices of
        # the same dimension, which makes it possible to export it to ONNX.
        # for MaxPool, GlobalMaxPool does not return indices,
        # so we try using max_poolxd_with_indices, and if it is not possible
        # (input is not a complete tensor or output size not factor of input size)
        # then we call GlobalAveragePool and return None for the indices
        output_size_value = output_size
        try:
            output_size = symbolic_helper._parse_arg(output_size, "is")
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            return symbolic_helper._onnx_unsupported(
                "adaptive pooling, since output_size is not constant.", input
            )
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        sizes = symbolic_helper._get_tensor_sizes(input)
        try:
            dim = sizes[2:]
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            dim = None
        if dim is None or any(i is None for i in dim):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "input size not accessible", input
            )
        # verify if output size % input size = 0 for all dim
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "output size that are not factor of input size", output_size_value
            )
        k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
        # call max_poolxd_with_indices to get indices in the output
        if type == "MaxPool":
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        output = g.op(type, input, kernel_shape_i=tuple_fn(k), strides_i=tuple_fn(k))
        return output

    return symbolic_fn


@_beartype.beartype
def _prepare_onnx_paddings(dim: int, pad):
    """Generate paddings in ONNX order based on pad in pytorch.
    Args:
        dim: the dimension of the tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ...
    """
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    return paddings


@_beartype.beartype
def _convert_padding_node(input):
    padding = symbolic_helper._maybe_get_const(input, "is")
    if symbolic_helper._is_value(padding) and symbolic_helper._is_packed_list(padding):
        input_list = symbolic_helper._unpack_list(padding)
        try:
            padding = [
                symbolic_helper._get_const(v, "i", "padding") for v in input_list
            ]
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "Pad", 9, 11, "The sizes of the padding must be constant", input
            )
    return padding


@_onnx_symbolic("aten::constant_pad_nd")
@_beartype.beartype
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value):
    mode = "constant"
    try:
        value = symbolic_helper._get_const(value, "f", "value")
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Pad", 9, 11, "The value for the padding must be constant", value
        )

    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, value_f=value, opset_before=11
    )


@_beartype.beartype
def _pad_circular(g: jit_utils.GraphContext, input: _C.Value, pad: _C.Value):
    padding = _convert_padding_node(pad)
    assert len(padding) % 2 == 0
    ndim = len(padding) // 2

    cur = input
    for idx in range(ndim):
        pad_r = padding[-(2 * idx + 1)]
        pad_l = padding[-(2 * idx + 2)]
        tensors = []
        if pad_l > 0:
            left = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[-(pad_l)], ends=[_constants.INT64_MAX]
            )
            tensors.append(left)

        if pad_l < 0 or pad_r < 0:
            start = builtins.max(0, -pad_l)
            end = -(builtins.max(0, -pad_r))
            middle = symbolic_helper._slice_helper(
                g,
                cur,
                axes=[2 + idx],
                starts=[start],
                ends=[end],
            )
            tensors.append(middle)
        else:
            tensors.append(cur)

        if pad_r > 0:
            right = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[0], ends=[pad_r]
            )
            tensors.append(right)

        cur = g.op("Concat", *tensors, axis_i=(2 + idx))

    return cur


@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
@_beartype.beartype
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    mode = "reflect"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
@_beartype.beartype
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = "edge"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


@_onnx_symbolic("aten::pad")
@_beartype.beartype
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    mode = symbolic_helper._parse_arg(mode, "s")
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        return _pad_circular(g, input, pad)
    else:
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest"),
        _export("upsample_nearest1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest"),
        _export("upsample_nearest2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest"),
        _export("upsample_nearest3d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[
        symbolic_helper._apply_params("upsample_linear1d", 3, "linear"),
        _export("upsample_linear1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[
        symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear"),
        _export("upsample_bilinear2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[
        symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear"),
        _export("upsample_trilinear3d"),
    ],
)
@_beartype.beartype
def _interpolate(name: str, dim: int, interpolate_mode: str):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
@_beartype.beartype
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Upsample", input, scales, mode_s=mode)


@_onnx_symbolic("aten::bitwise_not")
@_beartype.beartype
def bitwise_not(g: jit_utils.GraphContext, input):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            input,
        )
    return g.op("Not", input)


@_onnx_symbolic("aten::bitwise_or")
@_beartype.beartype
def bitwise_or(g, self, other):
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. self: ",
            self,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. other: ",
            other,
        )
    return g.op("Or", self, other)


@_beartype.beartype
def wrap_logical_op_with_cast_to(to_type):
    def decorator(fn):
        @functools.wraps(fn)
        def wrap_with_cast(g, input, other):
            to_cast_func = globals()[f"_cast_{to_type}"]
            return fn(g, to_cast_func(g, input, False), to_cast_func(g, other, False))

        return wrap_with_cast

    return decorator


@_beartype.beartype
def wrap_logical_op_with_negation(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrap_with_not(g, input, other):
        return g.op("Not", func(g, input, other))

    return wrap_with_not


@_onnx_symbolic("aten::__not_")
@_beartype.beartype
def __not_(g: jit_utils.GraphContext, self):
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            self,
        )
    return g.op("Not", self)


@_onnx_symbolic("aten::eq")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def eq(g: jit_utils.GraphContext, self, other):
    if isinstance(self.type(), _C.DeviceObjType) and isinstance(
        other.type(), _C.DeviceObjType
    ):
        # ONNX doesn't have devices, so consider them all to be equal.
        # The no-op check for equality will get constant-folded.
        return g.op("Constant", value_t=torch.tensor(True, dtype=torch.bool))
    self_node = self.node()
    other_node = other.node()
    if self_node.kind() == other_node.kind() == "onnx::Constant":
        if self_node.kindOf("value") == other_node.kindOf("value") == "s":
            # Exporting strings to ONNX is not supported.
            # If both strings are constant, we can compare them directly.
            # The no-op check for equality will get constant-folded.
            return g.op(
                "Constant",
                value_t=torch.tensor(
                    self_node.s("value") == other_node.s("value"),
                    dtype=torch.bool,
                ),
            )

    return g.op("Equal", self, other)


@_onnx_symbolic("aten::ne")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
@_beartype.beartype
def ne(g: jit_utils.GraphContext, self, other):
    return eq(g, self, other)


@_onnx_symbolic("aten::gt")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def gt(g: jit_utils.GraphContext, input, other):
    return _gt_impl(g, input, other)


@_beartype.beartype
def _gt_impl(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op("Greater", input, other)


@_onnx_symbolic("aten::lt")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def lt(g: jit_utils.GraphContext, input, other):
    return _lt_impl(g, input, other)


@_beartype.beartype
def _lt_impl(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op("Less", input, other)


@_onnx_symbolic("aten::ge")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
@_beartype.beartype
def ge(g: jit_utils.GraphContext, input, other):
    return _lt_impl(g, input, other)


@_onnx_symbolic("aten::le")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
@_beartype.beartype
def le(g: jit_utils.GraphContext, input, other):
    return _gt_impl(g, input, other)


@_onnx_symbolic("aten::__and_")
@_beartype.beartype
def __and_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            other,
        )
    return g.op("And", input, other)


@_onnx_symbolic("aten::__or_")
@_beartype.beartype
def __or_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            other,
        )
    return g.op("Or", input, other)


@_onnx_symbolic("aten::__xor_")
@_beartype.beartype
def __xor_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            other,
        )
    return g.op("Xor", input, other)


@_onnx_symbolic("aten::logical_and")
@wrap_logical_op_with_cast_to("Bool")
@_beartype.beartype
def logical_and(g: jit_utils.GraphContext, input, other):
    return g.op("And", input, other)


@_onnx_symbolic("aten::logical_or")
@wrap_logical_op_with_cast_to("Bool")
@_beartype.beartype
def logical_or(g: jit_utils.GraphContext, input, other):
    return g.op("Or", input, other)


@_onnx_symbolic("aten::logical_xor")
@wrap_logical_op_with_cast_to("Bool")
@_beartype.beartype
def logical_xor(g: jit_utils.GraphContext, input, other):
    return g.op("Xor", input, other)


@_onnx_symbolic("aten::logical_not")
@_beartype.beartype
def logical_not(g: jit_utils.GraphContext, input):
    return g.op("Not", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.BOOL))


@_onnx_symbolic("aten::__rshift_")
@_beartype.beartype
def __rshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    rshift = g.op("Div", self, two_pow)
    return rshift


@_onnx_symbolic("aten::__lshift_")
@_beartype.beartype
def __lshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
@_beartype.beartype
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if not symbolic_helper._is_bool(condition):
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    if self is None:
        condition = nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    return g.op("Where", condition, self, other)


@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # PyTorch dim and ONNX axis have different meanings.
    # See Softmax comment for details.
    # TODO: remove this as onnx opset 11 spec allows negative axes
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )
    if dim < 0:
        dim = input_dim + dim
    is_transpose_required = input_dim != dim + 1
    # ONNX only supports log_softmax with dim = -1. Transpose must be added before and after log_softmax to support other cases.
    if is_transpose_required:
        axes = list(range(input_dim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        input = g.op("Transpose", input, perm_i=axes)
        dim = input_dim - 1
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    if is_transpose_required:
        return_op = g.op("Transpose", return_op, perm_i=axes)  # type: ignore[possibly-undefined]
    return return_op


@_onnx_symbolic("aten::_log_softmax")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def _log_softmax(g: jit_utils.GraphContext, input, dim, half_to_float):
    if (
        half_to_float
        and _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.UNDEFINED
        )
        == _type_utils.JitScalarType.HALF
    ):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return log_softmax(g, input, dim)


@_onnx_symbolic("aten::_convolution")
@symbolic_helper.parse_args(
    "v", "v", "v", "is", "is", "is", "i", "is", "i", "i", "i", "i", "i"
)
@_beartype.beartype
def _convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        kernel_shape = None

    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    args = [input, weight]
    # ONNX only supports 1D bias
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        args.append(bias)

    kwargs = {
        "kernel_shape_i": weight_size[2:],
        "strides_i": stride,
        # NB: ONNX supports asymmetric padding, whereas PyTorch supports only
        # symmetric padding
        "pads_i": padding + padding,
        "dilations_i": dilation,
        "group_i": groups,
    }

    if any(o != 0 for o in output_padding):
        # ONNX supports both output_shape and output_padding. they are equivalent expressive.
        # output_padding is more straightforward, so we use it here.
        # output_shape = stride * (input_shape - 1) + output_padding + kernel_shape - padding * 2
        assert transposed
        assert len(stride) == len(output_padding)
        kwargs["output_padding_i"] = output_padding

    n = g.op("ConvTranspose" if transposed else "Conv", *args, **kwargs)

    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        return g.op("Add", n, bias)
    else:
        return n


@_onnx_symbolic("aten::_convolution_mode")
@symbolic_helper.parse_args(
    "v",
    "v",
    "v",
    "is",
    "s",
    "is",
    "i",
)
@_beartype.beartype
def _convolution_mode(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        kernel_shape = None

    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    args = [input, weight]
    # ONNX only supports 1D bias
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        args.append(bias)

    if padding == "valid":
        padding = "VALID"
    elif padding == "same":
        padding = "SAME_UPPER"
    kwargs = {
        "kernel_shape_i": weight_size[2:],
        "strides_i": stride,
        "auto_pad_s": padding,
        "dilations_i": dilation,
        "group_i": groups,
    }

    n = g.op("Conv", *args, **kwargs)

    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        return g.op("Add", n, bias)
    else:
        return n


@_onnx_symbolic("aten::convolution")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is", "i")
@_beartype.beartype
def convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
@_beartype.beartype
def conv1d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
@_beartype.beartype
def conv2d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
@_beartype.beartype
def conv3d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv_transpose1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose1d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv_transpose2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv_transpose3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose3d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
@_beartype.beartype
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):
    symbolic_helper.check_training_mode(training, "batch_norm")

    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            9,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
            input,
        )

    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        outputs=1 if not training else 5,
    )
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        return res


@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
@_beartype.beartype
def native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> Tuple[_C.Value, _C.Value, _C.Value]:
    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = symbolic_helper._generate_wrapped_number(g, 2.0)
    eps_cst = symbolic_helper._generate_wrapped_number(g, eps)

    if g.opset < 18:
        mean = g.op("ReduceMean", input, axes_i=axes)
    else:
        mean = g.op(
            "ReduceMean",
            input,
            g.op("Constant", value_t=torch.tensor(axes, dtype=torch.long)),
        )

    numerator = sub(g, input, mean)

    # Cast it to eps dtype to avoid precision loss
    is_type_half = (
        _type_utils.JitScalarType.from_value(numerator)
        == _type_utils.JitScalarType.HALF
    )
    if is_type_half:
        eps_dtype = _type_utils.JitScalarType.from_value(eps_cst)
        numerator = g.op(
            "Cast", numerator, to_i=_type_utils.JitScalarType(eps_dtype).onnx_type()
        )

    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    if g.opset < 18:
        variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    else:
        variance = g.op(
            "ReduceMean",
            pow(g, numerator, two_cst),
            g.op("Constant", value_t=torch.tensor(axes, dtype=torch.long)),
        )

    denominator = sqrt(g, g.op("Add", variance, eps_cst))
    normalized = g.op("Div", numerator, denominator)

    # Cast back to input type as eps related ops are all done
    if is_type_half:
        input_dtype = _type_utils.JitScalarType.from_value(input)
        normalized = g.op(
            "Cast", normalized, to_i=_type_utils.JitScalarType(input_dtype).onnx_type()
        )

    if not (weight is None or symbolic_helper._is_none(weight)):
        normalized = mul(g, normalized, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        normalized = add(g, normalized, bias)

    # rdenominator := 1 / sqrt(variance + eps)
    # According to aten::native_layer_norm, rdenominator should have the same dtype as input,
    # mean and normalized, so we need to Cast it back
    if is_type_half:
        denominator = g.op(
            "Cast", denominator, to_i=_type_utils.JitScalarType(input_dtype).onnx_type()  # type: ignore[possibly-undefined]
        )
        rdenominator = g.op("Reciprocal", denominator)
    else:
        rdenominator = reciprocal(g, denominator)

    return normalized, mean, rdenominator


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
@_beartype.beartype
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
) -> _C.Value:
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "layer_norm",
            input,
            weight,
            bias,
            normalized_shape_i=normalized_shape,
            eps_f=eps,
            cudnn_enable_i=cudnn_enable,
        )
    normalized, _, _ = native_layer_norm(g, input, normalized_shape, weight, bias, eps)
    return normalized


@_onnx_symbolic("aten::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "b", "f", "f", "b")
@_beartype.beartype
def instance_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats: bool,
    momentum: Number,
    eps: Number,
    cudnn_enabled: bool,
):
    symbolic_helper.check_training_mode(use_input_stats, "instance_norm")
    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if weight is None or symbolic_helper._is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or symbolic_helper._is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)
    if (
        running_mean is None
        or symbolic_helper._is_none(running_mean)
        or running_var is None
        or symbolic_helper._is_none(running_var)
    ):
        return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)
    else:
        input_size = symbolic_helper._get_tensor_sizes(input)
        # If input shape is [N, C, H, W], reshape to [1, N * C, H, W] and call batch_norm.
        # For more information instance_norm():
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L542
        input_size_reshape = input_size.copy()
        n = input_size[0]
        if n is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm training for unknown "
                "batch size.",
                input,
            )
        c = input_size[1]
        input_size_reshape[0] = 1
        input_size_reshape[1] = n * c
        weight_ = repeat(
            g, weight, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        bias_ = repeat(
            g, bias, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        running_mean_ = repeat(
            g,
            running_mean,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        running_var_ = repeat(
            g,
            running_var,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        input_reshaped = g.op(
            "Reshape",
            input,
            g.op("Constant", value_t=torch.LongTensor(input_size_reshape)),
        )
        out = batch_norm(
            g,
            input_reshaped,
            weight_,
            bias_,
            running_mean_,
            running_var_,
            use_input_stats,
            momentum,
            eps,
            cudnn_enabled,
        )
        return view(g, out, g.op("Constant", value_t=torch.tensor(input_size)))


@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("unfold", input, dimension_i=dimension, size_i=size, step_i=step)
    sizes = symbolic_helper._get_tensor_sizes(input)
    # FIXME(justinchuby): Get rid of the try catch here to improve readability
    try:
        sizedim = sizes[dimension]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        sizedim = None
    if sizedim is not None:
        low_indices = range(0, sizedim, step)
        hi_indices = range(size, sizedim + 1, step)
        stack = [
            symbolic_helper._slice_helper(
                g, input, axes=[dimension], starts=[low], ends=[hi]
            )
            for low, hi in zip(low_indices, hi_indices)
        ]
        ndim = len(sizes)
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))
        unsqueeze = [
            symbolic_helper._unsqueeze_helper(
                g, g.op("Transpose", t, perm_i=perm), [dimension]
            )
            for t in stack
        ]
        return g.op("Concat", *unsqueeze, axis_i=dimension)
    else:
        return symbolic_helper._unimplemented(
            "Unfold", "input size not accessible", input
        )


@_onnx_symbolic("aten::elu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "t", "t", "t")
@_beartype.beartype
def elu(g: jit_utils.GraphContext, input, alpha, scale, input_scale):
    if scale and scale != 1.0:
        return symbolic_helper._unimplemented(
            "scale", "does not support scale in Elu", scale
        )
    if input_scale and input_scale != 1.0:
        return symbolic_helper._unimplemented(
            "input_scale", "does not support input_scale in Elu", input_scale
        )
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=symbolic_helper._scalar(alpha))


@_onnx_symbolic("aten::selu")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def selu(g: jit_utils.GraphContext, input):
    return g.op("Selu", input)


@_onnx_symbolic("aten::index_select")
@symbolic_helper.parse_args("v", "i", "v")
@_beartype.beartype
def index_select(g: jit_utils.GraphContext, self, dim, index):
    # In case of a scalar index, index_select returns a tensor with the same rank as the input.
    # To match this behavior in ONNX, we make index a 1D tensor so that the following gather
    # also produces a tensor with the same rank as the input.
    return symbolic_helper._select_helper(g, self, dim, index)


@_onnx_symbolic("aten::index_put")
@_beartype.beartype
def index_put(g: jit_utils.GraphContext, self, indices_list_value, values, accumulate):
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    if symbolic_helper.is_caffe2_aten_fallback():
        args = [self] + indices_list + [values, accumulate]
        return g.at("index_put", *args)

    accumulate = symbolic_helper._parse_arg(accumulate, "b")

    if len(indices_list) == 0:
        if accumulate:
            return add(g, self, values)
        return values
    symbolic_helper._onnx_opset_unsupported("index_put", 9, 11, self)


@_onnx_symbolic("aten::index_fill")
@_beartype.beartype
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    dim_value = symbolic_helper._parse_arg(dim, "i")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "index_fill",
            self,
            index,
            value,
            overload_name="int_Scalar",
            dim_i=dim_value,
        )

    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    expanded_value = expand(g, value, expanded_index_shape, None)

    return scatter(g, self, dim, expanded_index, expanded_value)


@_onnx_symbolic("aten::index_copy")
@_beartype.beartype
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    dim_value = symbolic_helper._parse_arg(dim, "i")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index_copy", self, index, source, dim_i=dim_value)
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@_onnx_symbolic("aten::bucketize")
@symbolic_helper.parse_args("v", "v", "b", "b")
@_beartype.beartype
def bucketize(
    g: jit_utils.GraphContext, self, boundaries, out_int32=False, right=False
):
    out_type = _C_onnx.TensorProtoDataType.INT64
    if out_int32:
        out_type = _C_onnx.TensorProtoDataType.INT32
    # A tensor expanded_boundaries is created such that it
    # contains a copy of boundaries for each element of self.
    new_shape = g.op("Concat", g.op("Shape", boundaries), g.op("Shape", self), axis_i=0)
    # Unsqueeze step is performed to respect ONNX's numpy style broadcasting for comparison ops
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    assert tensor_rank is not None
    unsqueeze_axes = list(range(1, tensor_rank + 1))
    expanded_boundaries = expand(
        g,
        symbolic_helper._unsqueeze_helper(g, boundaries, unsqueeze_axes),
        new_shape,
        None,
    )
    # Compare each element of self to boundaries to get a tensor
    # with leading 1s and trailing 0s.
    # e.g., 4 > [1, 3, 4] = [1, 1, 0]
    # The index of the last 1 is the bucket where the element should go.
    if right:
        cond = ge(g, self, expanded_boundaries)
    else:
        cond = gt(g, self, expanded_boundaries)
    cond_out = g.op("Cast", cond, to_i=out_type)
    # Sum to get the number of 1s corresponding to each element,
    # which is the same as the bucket index.
    # e.g., sum(4 > [1, 3, 4]) = sum([1, 1, 0]) = 2
    return symbolic_helper._reducesum_helper(g, cond_out, axes_i=[0], keepdims_i=0)


@_onnx_symbolic("aten::type_as")
@_beartype.beartype
def type_as(g: jit_utils.GraphContext, self, other):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    other_dtype = symbolic_helper._try_get_scalar_type(other)
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    if other_dtype is not None:
        return g.op(
            "Cast",
            self,
            to_i=other_dtype.onnx_type(),
        )

    if symbolic_helper.is_caffe2_aten_fallback():
        # We don't know the type of other, bail by emitting ATen
        return g.at("type_as", self, other)

    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of type_as for tensor "
        "of unknown dtype. Please check if the dtype of the "
        "parameter passed to the type_as function is correct.",
        other,
    )


@_onnx_symbolic("aten::cosine_similarity")
@symbolic_helper.parse_args("v", "v", "i", "f")
@_beartype.beartype
def cosine_similarity(g: jit_utils.GraphContext, x1, x2, dim, eps):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("cosine_similarity", x1, x2, dim_i=dim, eps_f=eps)
    cross = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x2), axes_i=[dim], keepdims_i=0
    )
    x1_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x1), axes_i=[dim], keepdims_i=0
    )
    x2_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x2, x2), axes_i=[dim], keepdims_i=0
    )
    div_tens = max(
        g, sqrt(g, mul(g, x1_l2, x2_l2)), g.op("Constant", value_t=torch.tensor([eps]))
    )
    return div(g, cross, div_tens)


@_onnx_symbolic("aten::pairwise_distance")
@_beartype.beartype
def pairwise_distance(g: jit_utils.GraphContext, input1, input2, p, eps, keepdim):
    if not symbolic_helper._is_value(eps):
        eps = g.op("Constant", value_t=torch.tensor([eps]))
    inv_p = div(
        g,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.float)),
        add(g, p, eps),
    )
    summation = symbolic_helper._reducesum_helper(
        g,
        pow(g, sub(g, input1, input2), p),
        axes_i=[-1],
        keepdims_i=symbolic_helper._parse_arg(keepdim, "i"),
    )
    return pow(g, summation, inv_p)


@_onnx_symbolic("aten::clone")
# ignore clone operators that are inserted by PyTorch autograd
@_beartype.beartype
def clone(g: jit_utils.GraphContext, input, unused_memory_format):
    return input


@_onnx_symbolic("aten::abs")
@_beartype.beartype
def abs(g: jit_utils.GraphContext, self):
    return g.op("Abs", self)


@_onnx_symbolic("aten::log")
@_beartype.beartype
def log(g: jit_utils.GraphContext, self):
    return g.op("Log", self)


@_onnx_symbolic("aten::log1p")
@_beartype.beartype
def log1p(g: jit_utils.GraphContext, self):
    return log(g, add(g, symbolic_helper._if_scalar_type_as(torch.ones(1), self), self))


@_onnx_symbolic("aten::log10")
@_beartype.beartype
def log10(g: jit_utils.GraphContext, self):
    _ln10 = 2.30258509299404568401
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor([_ln10])))


@_onnx_symbolic("aten::pow")
@_beartype.beartype
def pow(g: jit_utils.GraphContext, self, exponent):
    f_dtype = _type_utils.JitScalarType.from_value(self)
    if not symbolic_helper._is_fp(self):
        f_dtype = _type_utils.JitScalarType.FLOAT
        self = g.op("Cast", self, to_i=f_dtype.onnx_type())
    if not symbolic_helper._is_fp(exponent):
        exponent = g.op(
            "Cast",
            exponent,
            to_i=f_dtype.onnx_type(),
        )
    pow = g.op("Pow", self, exponent)
    return pow


@_onnx_symbolic("aten::clamp")
@_beartype.beartype
def clamp(g: jit_utils.GraphContext, self, min, max):
    # min or max may be None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax
    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if symbolic_helper._is_constant(min) and symbolic_helper._is_constant(max):
            return symbolic_helper._op_with_optional_float_cast(
                g,
                "Clip",
                self,
                min_f=symbolic_helper._parse_arg(min, "f"),
                max_f=symbolic_helper._parse_arg(max, "f"),
                opset_before=12,
            )
        else:
            return clamp_max(g, clamp_min(g, self, min), max)


@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def clamp_min(g: jit_utils.GraphContext, self, min):
    if symbolic_helper._is_constant(min):
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min_f=symbolic_helper._parse_arg(min, "f"), opset_before=12
        )
    else:
        dtype = _type_utils.JitScalarType.from_value(self)
        min = g.op("Cast", min, to_i=dtype.onnx_type())
        return symbolic_helper._op_with_optional_float_cast(
            g, "Max", self, min, opset_before=12
        )


@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def clamp_max(g: jit_utils.GraphContext, self, max):
    if symbolic_helper._is_constant(max):
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, max_f=symbolic_helper._parse_arg(max, "f"), opset_before=12
        )
    else:
        dtype = _type_utils.JitScalarType.from_value(self)
        max = g.op("Cast", max, to_i=dtype.onnx_type())
        return symbolic_helper._op_with_optional_float_cast(
            g, "Min", self, max, opset_before=12
        )


@_onnx_symbolic("aten::max")
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
# TODO(justinchuby): Support multiple quantized args in output
@_beartype.beartype
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._max_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::maximum")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def maximum(g: jit_utils.GraphContext, input, other):
    return max(g, input, dim_or_y=other)


@_onnx_symbolic("aten::min")
# TODO(justinchuby): Support multiple quantized args in output
@_beartype.beartype
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    return symbolic_helper._min_helper(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::minimum")
@symbolic_helper.quantized_args(True, True)
@_beartype.beartype
def minimum(g: jit_utils.GraphContext, input, other):
    return min(g, input, dim_or_y=other)


@_onnx_symbolic("aten::amax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def amax(g: jit_utils.GraphContext, self, dim, keepdim):
    return g.op("ReduceMax", self, axes_i=dim, keepdims_i=keepdim)


@_onnx_symbolic("aten::amin")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def amin(g: jit_utils.GraphContext, self, dim, keepdim):
    return g.op("ReduceMin", self, axes_i=dim, keepdims_i=keepdim)


@_onnx_symbolic("aten::aminmax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def aminmax(g: jit_utils.GraphContext, self, dim, keepdim):
    reduce_kwargs = {"keepdims_i": keepdim}
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        reduce_kwargs["axes_i"] = [dim]

    return g.op("ReduceMin", self, **reduce_kwargs), g.op(
        "ReduceMax", self, **reduce_kwargs
    )


@_onnx_symbolic("aten::exp")
@_beartype.beartype
def exp(g: jit_utils.GraphContext, self):
    return g.op("Exp", self)


@_onnx_symbolic("aten::dropout_")
@_onnx_symbolic("aten::dropout")
@symbolic_helper.parse_args("v", "f", "i")
@_beartype.beartype
def dropout(g: jit_utils.GraphContext, input, p, train):
    symbolic_helper.check_training_mode(train, "dropout")
    # if train is False, dropout is no-op
    if not train:
        return input
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


@_onnx_symbolic(
    "aten::alpha_dropout_",
    decorate=[symbolic_helper._apply_params("aten::alpha_dropout_")],
)  # See Note [Export inplace]
@_onnx_symbolic(
    "aten::feature_alpha_dropout_",
    decorate=[symbolic_helper._apply_params("aten::feature_alpha_dropout_")],
)
@_onnx_symbolic(
    "aten::feature_dropout_",
    decorate=[symbolic_helper._apply_params("aten::feature_dropout_")],
)
@_onnx_symbolic(
    "aten::feature_alpha_dropout",
    decorate=[symbolic_helper._apply_params("aten::feature_alpha_dropout")],
)
@_onnx_symbolic(
    "aten::alpha_dropout",
    decorate=[symbolic_helper._apply_params("aten::alpha_dropout")],
)
@_onnx_symbolic(
    "aten::feature_dropout",
    decorate=[symbolic_helper._apply_params("aten::feature_dropout")],
)
@_beartype.beartype
def _unsupported_dropout(name: str):
    @symbolic_helper.parse_args("v", "none", "b")
    @_beartype.beartype
    def feature_dropout(g, input, p, train):
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        if train:
            return symbolic_helper._unimplemented(name, "training mode", input)
        return input

    return feature_dropout


@_onnx_symbolic("aten::norm")
@symbolic_helper.parse_args("v", "t", "is", "i", "v")
@_beartype.beartype
def norm(g: jit_utils.GraphContext, self, p, dim, keepdim, dtype=None):
    if p == 1:
        f = symbolic_helper._reduce_op_symbolic_helper("ReduceL1")
    elif p == 2:
        f = symbolic_helper._reduce_op_symbolic_helper("ReduceL2")
    else:
        raise errors.SymbolicValueError(
            "ONNX export only p-norms with p of 1 or 2", self
        )
    result = f(g, self, dim=dim, keepdim=keepdim)
    if dtype is not None:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    return result


@_onnx_symbolic("aten::conv_tbc")
@symbolic_helper.parse_args("v", "v", "v", "i")
@_beartype.beartype
def conv_tbc(g: jit_utils.GraphContext, input, weight, bias, pad):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("conv_tbc", input, weight, bias, pad_i=pad)
    else:
        # input must have 3 dimensions, see:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ConvolutionTBC.cpp#L8-L10
        # input = (time, batch, in_channels)
        # weight = (kernel_width, in_channels, out_channels)
        # bias = (out_channels,)
        input = g.op("Transpose", input, perm_i=[1, 2, 0])
        weight = g.op("Transpose", weight, perm_i=[2, 1, 0])
        conv = conv1d(g, input, weight, bias, [1], [pad], [1], 1)
        return g.op("Transpose", conv, perm_i=[2, 0, 1])


@_onnx_symbolic("aten::_unique")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def _unique(g: jit_utils.GraphContext, input, sorted, return_inverse):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "_unique",
            input,
            sorted_i=sorted,
            return_inverse_i=return_inverse,
            outputs=2,
        )
    else:
        return symbolic_helper._onnx_unsupported("_unique", input)


@_onnx_symbolic("aten::_unique2")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def _unique2(g: jit_utils.GraphContext, input, sorted, return_inverse, return_counts):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "_unique2",
            input,
            sorted_i=sorted,
            return_inverse_i=return_inverse,
            return_counts_i=return_counts,
            outputs=3,
        )

    symbolic_helper._onnx_opset_unsupported("_unique2", 9, 11, input)


@_onnx_symbolic("aten::_cast_Byte")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Byte(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.UINT8)


@_onnx_symbolic("aten::_cast_Char")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Char(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT8)


@_onnx_symbolic("aten::_cast_Short")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Short(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT16)


@_onnx_symbolic("aten::_cast_Int")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Int(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)


@_onnx_symbolic("aten::_cast_Long")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Long(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT64)


@_onnx_symbolic("aten::_cast_Half")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Half(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT16)


@_onnx_symbolic("aten::_cast_Float")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Float(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT)


@_onnx_symbolic("aten::_cast_Double")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Double(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE)


@_onnx_symbolic("aten::_cast_Bool")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Bool(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.BOOL)


@_onnx_symbolic("aten::empty")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::empty_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::new_empty")
@_beartype.beartype
def new_empty(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    return empty(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::scalar_tensor")
@_beartype.beartype
def scalar_tensor(g: jit_utils.GraphContext, scalar, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = _type_utils.JitScalarType.FLOAT
    scalar = g.op("Cast", scalar, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    return scalar


@_onnx_symbolic("aten::tensor")
@_beartype.beartype
def tensor(
    g: jit_utils.GraphContext, data, dtype=None, device=None, requires_grad=False
):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if symbolic_helper._is_packed_list(data):
        if dtype is None:
            dtype = _type_utils.JitScalarType.from_value(
                symbolic_helper._unpack_list(data)[0]
            )
        input_list = list()
        for t in symbolic_helper._unpack_list(data):
            shape_reference = g.op("Constant", value_t=torch.LongTensor([1]))
            t = symbolic_helper._reshape_helper(g, t, shape_reference)
            t = g.op("Cast", t, to_i=_type_utils.JitScalarType(dtype).onnx_type())
            input_list.append(t)
        return g.op("Concat", *input_list, axis_i=0)
    else:
        if dtype is None:
            dtype = _type_utils.JitScalarType.from_value(data)
        if symbolic_helper._is_list(data) and (
            symbolic_helper._is_tensor_list(data)
            or symbolic_helper._is_scalar_list(data)
        ):
            data = g.op("ConcatFromSequence", data, axis_i=0, new_axis_i=1)
    return g.op("Cast", data, to_i=_type_utils.JitScalarType(dtype).onnx_type())


@_onnx_symbolic("aten::as_tensor")
@_beartype.beartype
def as_tensor(g: jit_utils.GraphContext, data, dtype=None, device=None):
    return tensor(g, data, dtype, device)


@_onnx_symbolic("aten::zeros")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
@_beartype.beartype
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device, layout and pin_memory in ONNX, so we ignore it
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )


@_onnx_symbolic("aten::zeros_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    shape = g.op("Shape", input)
    if symbolic_helper._is_none(dtype):
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )


@_onnx_symbolic("aten::new_zeros")
@_beartype.beartype
def new_zeros(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    self_dtype = symbolic_helper._try_get_scalar_type(self)

    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::zero")
@_beartype.beartype
def zero(g: jit_utils.GraphContext, self):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    return zeros_like(g, self, self_dtype)


@_onnx_symbolic("aten::ones")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
@_beartype.beartype
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor([1], dtype=scalar_type.dtype()),
    )


@_onnx_symbolic("aten::ones_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    shape = g.op("Shape", input)
    if symbolic_helper._is_none(dtype):
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([1], dtype=scalar_type.dtype()),
    )


@_onnx_symbolic("aten::new_ones")
@_beartype.beartype
def new_ones(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    return ones(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::full")
@_beartype.beartype
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=False
):
    const_value = symbolic_helper._maybe_get_const(value, "t")
    if symbolic_helper._is_value(const_value):
        dtype = _type_utils.JitScalarType.FLOAT if dtype is None else dtype
        tmp = zeros(g, sizes, dtype, layout, device)
        return add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        if dtype is None:
            scalar_type = _type_utils.JitScalarType.FLOAT
        else:
            scalar_type = _type_utils.JitScalarType(dtype)
        sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
        if isinstance(sizes_, list) and len(sizes_) == 0:
            sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
        return g.op(
            "ConstantOfShape",
            sizes,
            value_t=const_value.view(1).to(scalar_type.dtype()),
        )


@_onnx_symbolic("aten::full_like")
@_beartype.beartype
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    fill_value = symbolic_helper._maybe_get_const(fill_value, "f")
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    if symbolic_helper._is_value(fill_value):
        tmp = zeros_like(g, input, dtype, layout, device)
        fill_value = g.op("Cast", fill_value, to_i=scalar_type.onnx_type())
        return add(g, tmp, fill_value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        shape = g.op("Shape", input)
        return g.op(
            "ConstantOfShape",
            shape,
            value_t=torch.tensor([fill_value], dtype=scalar_type.dtype()),
        )


@_onnx_symbolic("aten::new_full")
@_beartype.beartype
def new_full(
    g: jit_utils.GraphContext,
    self,
    size,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=False,
):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    return full(g, size, fill_value, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::eye")
@_beartype.beartype
def eye(g: jit_utils.GraphContext, *args):
    if len(args) == 5:
        # aten::eye(n, dtype, layout, device, pin_memory)
        n, dtype, layout, device, pin_memory = args
        dim_size = symbolic_helper._unsqueeze_helper(g, n, [0])
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    if len(args) == 6:
        # aten::eye(n, m, dtype, layout, device, pin_memory)
        n, m, dtype, layout, device, pin_memory = args
        shape = g.op(
            "Concat",
            symbolic_helper._unsqueeze_helper(g, n, [0]),
            symbolic_helper._unsqueeze_helper(g, m, [0]),
            axis_i=0,
        )
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)

    return symbolic_helper._unimplemented("aten::eye", f"with {len(args)} arguments")


@_onnx_symbolic("aten::slice")
@_beartype.beartype
def slice(g: jit_utils.GraphContext, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        dim, start, end, step = args
        step = symbolic_helper._parse_arg(step, "i")
        if step != 1:
            raise errors.SymbolicValueError("step!=1 is currently not supported", self)
        is_start_none = start.node().kind() == "prim::Constant" and isinstance(
            start.type(), _C.NoneType
        )
        is_end_none = end.node().kind() == "prim::Constant" and isinstance(
            end.type(), _C.NoneType
        )
        is_start_onnx_const = start.node().kind() == "onnx::Constant"
        is_end_onnx_const = end.node().kind() == "onnx::Constant"
        if (
            ((not is_start_none) and (not is_start_onnx_const))
            or ((not is_end_none) and (not is_end_onnx_const))
            or dim.node().kind() != "onnx::Constant"
        ):
            if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
                raise errors.SymbolicValueError(
                    "Unsupported: ONNX export of Slice with dynamic inputs. DynamicSlice "
                    "is a deprecated experimental op. Please use statically allocated "
                    "variables or export to a higher opset version.",
                    self,
                )
            else:
                start_unsqueezed = symbolic_helper._unsqueeze_helper(g, start, [0])
                end_unsqueezed = symbolic_helper._unsqueeze_helper(g, end, [0])
                dim_unsqueezed = symbolic_helper._unsqueeze_helper(g, dim, [0])
                return g.op(
                    "DynamicSlice",
                    self,
                    start_unsqueezed,
                    end_unsqueezed,
                    dim_unsqueezed,
                )
        else:
            start = 0 if is_start_none else symbolic_helper._parse_arg(start, "i")
            end = (
                _constants.INT64_MAX
                if is_end_none
                else symbolic_helper._parse_arg(end, "i")
            )
            dim = symbolic_helper._parse_arg(dim, "i")
            return symbolic_helper._slice_helper(
                g, self, axes=[dim], starts=[start], ends=[end]
            )
    elif len(args) == 3:
        # aten::slice(t[] l, int start, int end, int step) -> t[]
        start, end, step = args
        dim = 0
        is_start_none = start.node().kind() == "prim::Constant" and isinstance(
            start.type(), _C.NoneType
        )
        is_end_none = end.node().kind() == "prim::Constant" and isinstance(
            end.type(), _C.NoneType
        )
        start = 0 if is_start_none else symbolic_helper._parse_arg(start, "i")
        end = (
            _constants.INT64_MAX
            if is_end_none
            else symbolic_helper._parse_arg(end, "i")
        )
        return symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[start], ends=[end]
        )

    return symbolic_helper._unimplemented("aten::slice", f"with {len(args)} arguments")


@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
@_beartype.beartype
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    return symbolic_helper._op_with_optional_float_cast(
        g, "Clip", self, min_f=min_val, max_f=max_val, opset_before=12
    )


@_onnx_symbolic("aten::hardswish")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v")
@_beartype.beartype
def hardswish(g: jit_utils.GraphContext, self):
    hs = hardsigmoid(g, self)
    return g.op("Mul", self, hs)


@_onnx_symbolic("aten::hardsigmoid")
# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
@symbolic_helper.parse_args("v")
@_beartype.beartype
def hardsigmoid(g: jit_utils.GraphContext, self):
    # Set alpha_f to 1 / 6 to make op equivalent to PyTorch's definition of Hardsigmoid.
    # See https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    return g.op("HardSigmoid", self, alpha_f=1 / 6)


@_onnx_symbolic("aten::tanhshrink")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def tanhshrink(g: jit_utils.GraphContext, self):
    return g.op("Sub", self, tanh(g, self))


@_onnx_symbolic("aten::hardshrink")
@symbolic_helper.parse_args("v", "f")
@_beartype.beartype
def hardshrink(g: jit_utils.GraphContext, self, lambd):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    lambd_op = g.op(
        "Constant",
        value_t=torch.tensor(lambd, dtype=scalar_type.dtype()),
    )
    cond = logical_or(g, gt(g, self, lambd_op), lt(g, self, neg(g, lambd_op)))
    return g.op(
        "Where",
        cond,
        self,
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )


@_onnx_symbolic("aten::softshrink")
@symbolic_helper.parse_args("v", "f")
@_beartype.beartype
def softshrink(g: jit_utils.GraphContext, self, lambd):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    lambd_op = g.op(
        "Constant",
        value_t=torch.tensor(lambd, dtype=scalar_type.dtype()),
    )
    gt_cond = gt(g, self, lambd_op)
    gt_out = g.op(
        "Where",
        gt_cond,
        sub(g, self, lambd_op),
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )
    lt_cond = lt(g, self, neg(g, lambd_op))
    lt_out = g.op(
        "Where",
        lt_cond,
        add(g, self, lambd_op),
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )
    return add(g, gt_out, lt_out)


@_onnx_symbolic("aten::alias")
@_beartype.beartype
def alias(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("aten::unsqueeze")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    """Implement unsqueezing a pytorch tensor in ONNX by inserting a new dimension at the specified `dim`"""
    # Handle negative dim
    if dim < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            warnings.warn(
                "ONNX export unsqueeze with negative axis "
                + str(dim)
                + " might cause the onnx model to be incorrect. "
                + "Negative axis is not supported in ONNX. "
                + "Axis is converted to "
                + str(dim + rank + 1)
                + " based on input shape at export time. "
                + "Passing an tensor of different rank in execution will be incorrect."
            )
            dim = dim + rank + 1
        else:
            return symbolic_helper._unimplemented(
                "unsqueeze", "negative axis with unknown input rank", self
            )

    return symbolic_helper._unsqueeze_helper(g, self, axes_i=[dim])


@_onnx_symbolic("aten::sort")
# TODO(justinchuby): Support multiple quantized args in output
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def sort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Sort", "Out parameter is not supported for sort", self
        )
    self_sizes = symbolic_helper._get_tensor_sizes(self)
    try:
        dim_size = self_sizes[dim]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        dim_size = None

    if dim_size is None:
        return symbolic_helper._unimplemented("Sort", "input size not accessible", self)

    return g.op("TopK", self, k_i=dim_size, axis_i=dim, outputs=2)


@_onnx_symbolic("aten::numel")
@_beartype.beartype
def numel(g: jit_utils.GraphContext, self):
    return symbolic_helper._numel_helper(g, self)


@_onnx_symbolic("aten::topk")
# TODO(justinchuby): Support multiple quantized args in output
@symbolic_helper.parse_args("v", "i", "i", "i", "i", "none")
@_beartype.beartype
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "TopK", "Out parameter is not supported for topk", self
        )
    if not largest:
        symbolic_helper._unimplemented("TopK", "Ascending TopK is not supported", self)

    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


@_onnx_symbolic("prim::convert_element_type")
@_beartype.beartype
def convert_element_type(g: jit_utils.GraphContext, self, *args):
    dtype = symbolic_helper._get_const(args[0], "i", "dtype")
    return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())


@_onnx_symbolic("aten::to")
@_beartype.beartype
def to(g: jit_utils.GraphContext, self, *args):
    @_beartype.beartype
    def is_aten_to_device_only(args):
        if len(args) == 4:
            # aten::to(Tensor, Device, bool, bool, memory_format)
            return (
                args[0].node().kind() == "prim::device"
                or args[0].type().isSubtypeOf(_C.ListType.ofInts())
                or isinstance(args[0].type(), _C.DeviceObjType)
            )
        elif len(args) == 5:
            # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
            # When dtype is None, this is a aten::to(device) call
            dtype = symbolic_helper._get_const(args[1], "i", "dtype")
            return dtype is None
        elif len(args) in (6, 7):
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) -> Tensor
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) -> Tensor
            # When dtype is None, this is a aten::to(device) call
            dtype = symbolic_helper._get_const(args[0], "i", "dtype")
            return dtype is None
        return False

    # ONNX doesn't have a concept of a device, so we ignore device-only casts
    if is_aten_to_device_only(args):
        return self

    if len(args) == 4:
        # TestONNXRuntime::test_ones_bool shows args[0] of aten::to() can be onnx::Constant[value=<Tensor>]()
        # In this case, the constant value is a tensor not int,
        # so symbolic_helper._maybe_get_const(args[0], 'i') would not work.
        dtype = args[0]
        if (
            symbolic_helper._is_value(args[0])
            and args[0].node().kind() == "onnx::Constant"
        ):
            tval = symbolic_helper._node_get(args[0].node(), "value")
            if isinstance(tval, torch.Tensor):
                if len(tval.shape) == 0:
                    tval = tval.item()
                    dtype = int(tval)
                else:
                    dtype = tval

        if symbolic_helper._is_value(dtype) or isinstance(dtype, torch.Tensor):
            # aten::to(Tensor, Tensor, bool, bool, memory_format)
            dtype = _type_utils.JitScalarType.from_value(args[0])
            return g.op(
                "Cast",
                self,
                to_i=dtype.onnx_type(),
            )
        else:
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            # memory_format is ignored
            return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        dtype = symbolic_helper._get_const(args[1], "i", "dtype")
        # memory_format is ignored
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    elif len(args) == 6:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) -> Tensor
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    elif len(args) == 7:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) -> Tensor
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())

    return symbolic_helper._onnx_unsupported("Unknown aten::to signature", self)


@_onnx_symbolic("aten::repeat")
@_beartype.beartype
def repeat(g: jit_utils.GraphContext, self, repeats):
    dtype = _type_utils.JitScalarType.INT64
    shape_ = ones_like(g, repeats, dtype)
    self = g.op("Expand", self, shape_)
    return g.op("Tile", self, repeats)


@_onnx_symbolic("aten::repeat_interleave")
@_beartype.beartype
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=None, output_size=None
):
    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    input_sizes = symbolic_helper._get_tensor_sizes(self)
    if repeats_dim is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats rank.",
            self,
        )
    if repeats_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size.",
            self,
        )
    if input_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size.",
            self,
        )

    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if symbolic_helper._is_none(dim):
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = torch.tensor(0, dtype=torch.int64)
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)

    # Handle cases where dim is negative
    if dim < 0:
        dim += len(input_sizes)

    input_sizes_temp = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            input_sizes[idx], input_sizes_temp[idx] = 0, -1

    # Cases where repeats is an int or single value tensor
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
                self,
            )
        return symbolic_helper._repeat_interleave_single_value_repeat_helper(
            g, self, repeats, dim
        )

    # Cases where repeats is a 1 dim Tensor
    elif repeats_dim == 1:
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
                self,
            )
        if repeats_sizes[0] is None:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported for cases with dynamic repeats",
                self,
            )
        assert (
            repeats_sizes[0] == input_sizes[dim]
        ), "repeats must have the same size as input along dim"
        reps = repeats_sizes[0]
    else:
        raise errors.SymbolicValueError("repeats must be 0-dim or 1-dim tensor", self)

    final_splits = list()
    r_splits = symbolic_helper._repeat_interleave_split_helper(g, repeats, reps, 0)
    i_splits = symbolic_helper._repeat_interleave_split_helper(g, self, reps, dim)
    input_sizes[dim], input_sizes_temp[dim] = -1, 1
    for idx, r_split in enumerate(r_splits):
        i_split = unsqueeze(g, i_splits[idx], dim + 1)
        r_concat = [
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[: dim + 1])),
            r_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[dim + 1 :])),
        ]
        r_concat = g.op("Concat", *r_concat, axis_i=0)
        i_split = expand(g, i_split, r_concat, None)
        i_split = symbolic_helper._reshape_helper(
            g,
            i_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes)),
            allowzero=0,
        )
        final_splits.append(i_split)
    return g.op("Concat", *final_splits, axis_i=dim)


@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    dims = symbolic_helper._get_tensor_sizes(self)
    if len(dims) != 4:
        return symbolic_helper._unimplemented(
            "pixel_shuffle", "only support 4d input", self
        )
    if any(i is None for i in dims[1:]):
        after_view = symbolic_helper._reshape_helper(
            g,
            symbolic_helper._unsqueeze_helper(g, self, [2, 3]),
            g.op(
                "Constant",
                value_t=torch.tensor([0, -1, upscale_factor, upscale_factor, 0, 0]),
            ),
            allowzero=0,
        )
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
        # For dynamic input shapes, two reshapes are performed
        reshape_h = symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op("Constant", value_t=torch.tensor([0, 0, -1, 1, 0, 0])),
            allowzero=0,
        )
        reshape_w = symbolic_helper._reshape_helper(
            g,
            reshape_h,
            g.op("Constant", value_t=torch.tensor([0, 0, 0, 0, -1, 1])),
            allowzero=0,
        )
        return symbolic_helper._squeeze_helper(g, reshape_w, [3, 5])
    else:
        output_channel = dims[1] // upscale_factor // upscale_factor
        after_view = symbolic_helper._reshape_helper(
            g,
            self,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        upscale_factor,
                        upscale_factor,
                        dims[2],
                        dims[3],
                    ]
                ),
            ),
            allowzero=0,
        )
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
        return symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        dims[2] * upscale_factor,
                        dims[3] * upscale_factor,
                    ]
                ),
            ),
            allowzero=0,
        )


@_onnx_symbolic("aten::pixel_unshuffle")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def pixel_unshuffle(g: jit_utils.GraphContext, self, downscale_factor):
    dims = symbolic_helper._get_tensor_sizes(self)
    if len(dims) != 4:
        return symbolic_helper._unimplemented(
            "pixel_shuffle", "only support 4d input", self
        )
    if any(i is None for i in dims[1:]):
        # For dynamic input shapes, two reshapes are performed
        reshape_h = symbolic_helper._reshape_helper(
            g,
            symbolic_helper._unsqueeze_helper(g, self, [3]),
            g.op("Constant", value_t=torch.tensor([0, 0, -1, downscale_factor, 0])),
            allowzero=0,
        )
        reshape_w = symbolic_helper._reshape_helper(
            g,
            reshape_h,
            g.op("Constant", value_t=torch.tensor([0, 0, 0, 0, -1, downscale_factor])),
            allowzero=0,
        )
        after_transpose = g.op("Transpose", reshape_w, perm_i=[0, 1, 3, 5, 2, 4])
        final_reshape = symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op("Constant", value_t=torch.tensor([0, -1, 1, 1, 0, 0])),
            allowzero=0,
        )
        return symbolic_helper._squeeze_helper(g, final_reshape, [2, 3])
    else:
        output_channel = dims[1] * downscale_factor * downscale_factor
        after_view = symbolic_helper._reshape_helper(
            g,
            self,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        dims[1],
                        dims[2] // downscale_factor,
                        downscale_factor,
                        dims[3] // downscale_factor,
                        downscale_factor,
                    ]
                ),
            ),
            allowzero=0,
        )
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 3, 5, 2, 4])
        return symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        dims[2] // downscale_factor,
                        dims[3] // downscale_factor,
                    ]
                ),
            ),
            allowzero=0,
        )


@_beartype.beartype
def _generic_rnn(
    g: jit_utils.GraphContext,
    variant,
    input,
    initial_states,
    all_weights,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first=None,
    batch_sizes=None,
):
    warnings.warn(
        "Exporting a model to ONNX with a batch_size other than 1, "
        + "with a variable length with "
        + variant
        + " can cause an error "
        + "when running the ONNX model with a different batch size. "
        + "Make sure to save the model with a batch size of 1, "
        + "or define the initial states (h0/c0) as inputs of the model. "
    )

    onnxActivations = [
        "Relu",
        "Tanh",
        "Sigmoid",
        "Affine",
        "LeakyRelu",
        "ThresholdedRelu",
        "ScaledTanh",
        "HardSigmoid",
        "Elu",
        "Softsign",
        "Softplus",
    ]
    variantToOnnxActivationMap = dict(
        zip([act_fun.lower() for act_fun in onnxActivations], onnxActivations)
    )
    weights_per_layer = 4 if has_biases else 2
    # this means that projections are used inside LSTM, so need to tell user that it's not supported
    if variant == "LSTM" and len(all_weights) != num_layers * weights_per_layer * (
        1 + bidirectional
    ):
        return symbolic_helper._unimplemented("LSTM", "LSTMs with projections", input)
    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [
        all_weights[i : i + weights_per_layer]
        for i in range(0, len(all_weights), weights_per_layer)
    ]
    if batch_first:
        # batch, seq, feat -> seq, batch, feat
        input = g.op("Transpose", input, perm_i=[1, 0, 2])
    if dropout and train:
        return symbolic_helper._unimplemented(
            "RNN/GRU/LSTM", "dropout in training mode", input
        )

    if variant.startswith("RNN"):
        nonlinearity = variantToOnnxActivationMap[variant[4:].lower()]
        variant = "RNN"

    w_hh = all_weights[1]
    hidden_size = symbolic_helper._get_tensor_dim_size(w_hh, 1)
    if hidden_size is None:
        return symbolic_helper._unimplemented(
            "RNN/GRU/LSTM", "unknown hidden size", input
        )

    unidirectional = not bidirectional

    prev_output = input

    h_outs = []
    if variant == "RNN" or variant == "GRU":
        h0 = initial_states
    elif variant == "LSTM":
        h0, c0 = initial_states
        c_outs = []

    sequence_lens = unused(g) if batch_sizes is None else batch_sizes

    if variant == "GRU":
        # pytorch is reset, input, hidden
        # onnx is    input, reset, hidden
        reform_permutation = [(1, 2), (0, 1), (2, 3)]
    elif variant == "LSTM":
        # pytorch is input, forget, cell, output.
        # onnx is    input, output, forget, cell.
        reform_permutation = [(0, 1), (3, 4), (1, 3)]

    @_beartype.beartype
    def reform_weights(g, w, n, intervals):
        slices = [
            symbolic_helper._slice_helper(g, w, axes=[0], starts=[x * n], ends=[y * n])
            for x, y in intervals
        ]
        return g.op("Concat", *slices, axis_i=0)

    @_beartype.beartype
    def transform_weights_no_bias(layer_index):
        weights = layer_weights[layer_index]
        if variant == "RNN":
            weight_ih, weight_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            weight_ih, weight_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh)  # type: ignore[possibly-undefined]
        )

    @_beartype.beartype
    def transform_weights(layer_index):
        weights = layer_weights[layer_index]
        if variant == "RNN":
            weight_ih, weight_hh, bias_ih, bias_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            weight_ih, weight_hh, bias_ih, bias_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        bias_concat = g.op("Concat", bias_ih, bias_hh, axis_i=0)  # type: ignore[possibly-undefined]
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0])
            for x in (weight_ih, weight_hh, bias_concat)  # type: ignore[possibly-undefined]
        )

    @_beartype.beartype
    def retrieve_state(x, start, end):
        return (
            x
            if num_layers == 1
            else symbolic_helper._slice_helper(
                g, x, axes=[0], starts=[start], ends=[end]
            )
        )

    for i in range(num_layers):
        if unidirectional:
            if weights_per_layer == 4:
                weight_ih, weight_hh, bias_concat = transform_weights(i)
            else:
                weight_ih, weight_hh = transform_weights_no_bias(i)
                bias_concat = unused(g)

            state_indices = i, i + 1
        else:
            if weights_per_layer == 4:
                weight_ih_f, weight_hh_f, bias_f = transform_weights(2 * i)
                weight_ih_b, weight_hh_b, bias_b = transform_weights(2 * i + 1)
                bias_concat = g.op("Concat", bias_f, bias_b, axis_i=0)
            else:
                weight_ih_f, weight_hh_f = transform_weights_no_bias(2 * i)
                weight_ih_b, weight_hh_b = transform_weights_no_bias(2 * i + 1)
                bias_concat = unused(g)

            weight_ih = g.op("Concat", weight_ih_f, weight_ih_b, axis_i=0)
            weight_hh = g.op("Concat", weight_hh_f, weight_hh_b, axis_i=0)

            state_indices = 2 * i, 2 * i + 2

        inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens]

        inputs.append(retrieve_state(h0, *state_indices))  # type: ignore[possibly-undefined]
        if variant == "LSTM":
            inputs.append(retrieve_state(c0, *state_indices))  # type: ignore[possibly-undefined]

        extra_kwargs = {} if unidirectional else {"direction_s": "bidirectional"}
        if variant == "RNN":
            if bidirectional:
                activation = [nonlinearity, nonlinearity]  # type: ignore[possibly-undefined]
            else:
                activation = [nonlinearity]  # type: ignore[possibly-undefined]

            prev_output, h_out = g.op(
                "RNN",
                *inputs,
                outputs=2,
                hidden_size_i=hidden_size,
                activations_s=activation,
                **extra_kwargs,
            )
        elif variant == "GRU":
            prev_output, h_out = g.op(
                "GRU",
                *inputs,
                outputs=2,
                hidden_size_i=hidden_size,
                linear_before_reset_i=1,
                **extra_kwargs,
            )
        elif variant == "LSTM":
            prev_output, h_out, c_out = g.op(
                "LSTM", *inputs, outputs=3, hidden_size_i=hidden_size, **extra_kwargs
            )

        if bidirectional:
            # The ONNX RNN/GRU/LSTM produce an output of dimensions
            #   seq_len, num_directions, batch, hidden_size
            # We have to convert to match pytorch's expected
            #   seq_len, batch, num_directions * hidden_size
            # by first moving num_directions before hidden_size with
            # Transpose, and then combining it with hidden_size
            # with Reshape.
            prev_output = g.op("Transpose", prev_output, perm_i=[0, 2, 1, 3])
            prev_output = symbolic_helper._reshape_helper(
                g,
                prev_output,
                g.op("Constant", value_t=torch.LongTensor([0, 0, -1])),
                allowzero=0,
            )
        else:
            prev_output = symbolic_helper._squeeze_helper(g, prev_output, [1])

        h_outs.append(h_out)  # type: ignore[possibly-undefined]
        if variant == "LSTM":
            c_outs.append(c_out)  # type: ignore[possibly-undefined]
    if batch_first:
        # seq, batch, num_directions * hidden_size -> batch, seq, num_directions * hidden_size
        prev_output = g.op("Transpose", prev_output, perm_i=[1, 0, 2])
    h_outs = h_out if num_layers == 1 else g.op("Concat", *h_outs, axis_i=0)  # type: ignore[possibly-undefined]
    if variant == "RNN" or variant == "GRU":
        return prev_output, h_outs
    elif variant == "LSTM":
        c_outs = c_out if num_layers == 1 else g.op("Concat", *c_outs, axis_i=0)  # type: ignore[possibly-undefined]
        return prev_output, h_outs, c_outs


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "f", "i", "i", "i")
@_beartype.beartype
def _lstm_full(
    g: jit_utils.GraphContext,
    input,
    hidden_v,
    weight_v,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    hidden, weight = symbolic_helper._unpack_list(
        hidden_v
    ), symbolic_helper._unpack_list(weight_v)
    return _generic_rnn(
        g,
        "LSTM",
        input,
        hidden,
        weight,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "f", "i", "i")
@_beartype.beartype
def _lstm_packed(
    g: jit_utils.GraphContext,
    input,
    batch_sizes,
    hidden_v,
    weight_v,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    hidden, weight = symbolic_helper._unpack_list(
        hidden_v
    ), symbolic_helper._unpack_list(weight_v)
    return _generic_rnn(
        g,
        "LSTM",
        input,
        hidden,
        weight,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_sizes=batch_sizes,
    )


@_onnx_symbolic("aten::lstm")
@_beartype.beartype
def lstm(g: jit_utils.GraphContext, *args):
    if symbolic_helper._is_tensor_list(args[3]):
        return _lstm_packed(g, *args)
    else:
        return _lstm_full(g, *args)


@_onnx_symbolic("aten::lstm_cell")
@_beartype.beartype
def lstm_cell(g: jit_utils.GraphContext, self, hidden, w_ih, w_hh, b_ih, b_hh):
    input = symbolic_helper._unsqueeze_helper(g, self, [0])
    hidden = symbolic_helper._unpack_list(hidden)
    hidden = [symbolic_helper._unsqueeze_helper(g, x, [0]) for x in hidden]
    weight = (
        (w_ih, w_hh, b_ih, b_hh) if symbolic_helper._is_tensor(b_ih) else (w_ih, w_hh)
    )
    has_biases = True if symbolic_helper._is_tensor(b_ih) else False
    _, h_outs, c_outs = _generic_rnn(
        g,
        "LSTM",
        input,
        hidden,
        weight,
        has_biases,
        num_layers=1,
        dropout=0,
        train=0,
        bidirectional=False,
        batch_first=False,
    )
    return symbolic_helper._squeeze_helper(
        g, h_outs, [0]
    ), symbolic_helper._squeeze_helper(g, c_outs, [0])


@_onnx_symbolic(
    "aten::gru", decorate=[symbolic_helper._apply_params("GRU"), _export("gru")]
)
@_onnx_symbolic(
    "aten::rnn_tanh",
    decorate=[symbolic_helper._apply_params("RNN_TANH"), _export("rnn_tanh")],
)
@_onnx_symbolic(
    "aten::rnn_relu",
    decorate=[symbolic_helper._apply_params("RNN_RELU"), _export("rnn_relu")],
)
def _one_hidden_rnn(kind: str):
    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "f", "i", "i", "i")
    @_beartype.beartype
    def _rnn_full(
        g,
        input,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        weight = symbolic_helper._unpack_list(weight_v)
        return _generic_rnn(
            g,
            kind,
            input,
            hidden,
            weight,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )

    @symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "f", "i", "i")
    def _rnn_packed(
        g,
        input,
        batch_sizes,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
    ):
        weight = symbolic_helper._unpack_list(weight_v)
        return _generic_rnn(
            g,
            kind,
            input,
            hidden,
            weight,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_sizes=batch_sizes,
        )

    def symbolic(g, *args):
        if symbolic_helper._is_tensor_list(args[3]):
            return _rnn_packed(g, *args)
        else:
            return _rnn_full(g, *args)

    return symbolic


@_onnx_symbolic("aten::_dim_arange")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    like_shape = g.op("Shape", like)
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.op("_caffe2::Range", stop)
    else:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        return arange(g, stop, 4, None, None, None)


@_onnx_symbolic("aten::detach")
@_beartype.beartype
def detach(g: jit_utils.GraphContext, input):
    # Erase aten::detach nodes because ONNX is inference only
    return input


@_onnx_symbolic("aten::contiguous")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def contiguous(g: jit_utils.GraphContext, input, memory_format):
    if memory_format > 2:  # allower values are any, preserve and contiguous_format
        raise errors.SymbolicValueError(
            "onnx memory_format support is not implemented", input
        )
    return input


@_onnx_symbolic("aten::_pack_padded_sequence")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def _pack_padded_sequence(g: jit_utils.GraphContext, input, lengths, batch_first):
    # Currently there is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.
    if batch_first:
        input = g.op("Transpose", input, perm_i=[1, 0, 2])
    if not lengths.type().isSubtypeOf(torch._C.TensorType.get()):
        raise errors.SymbolicValueError(
            "'lengths' must be a Tensor for ONNX export", input
        )
    # We know it's a TensorType so this check is now safe.
    # It's really only necessary because those operators expand to something that
    # only works with int32 types in Caffe2...
    if (
        _type_utils.JitScalarType.from_value(
            lengths, _type_utils.JitScalarType.UNDEFINED
        )
        != _type_utils.JitScalarType.INT
    ):
        lengths = g.op("Cast", lengths, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op("prim::PackPadded", input, lengths, outputs=2)


@_onnx_symbolic("aten::_pad_packed_sequence")
@symbolic_helper.parse_args("v", "v", "i", "t", "v")
@_beartype.beartype
def _pad_packed_sequence(
    g: jit_utils.GraphContext,
    data,
    batch_sizes,
    batch_first,
    padding_value,
    total_length,
):
    # Ignore total_length as it is not supported in _symbolic_pad_packed_sequence
    # It is only useful/used when training using data_parallel model, so
    # It shouldn't be relevant for ONNX anyway
    data, lengths = g.op("prim::PadPacked", data, batch_sizes, outputs=2)
    if batch_first:
        data = g.op("Transpose", data, perm_i=[1, 0, 2])
    return data, lengths


@_onnx_symbolic("aten::randint")
@_beartype.beartype
def randint(g: jit_utils.GraphContext, low, high, shapes, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    low_i = symbolic_helper._get_const(low, "i", "low")
    high_i = symbolic_helper._get_const(high, "i", "high")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.INT64
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    if low_i is None:
        raise symbolic_helper._onnx_unsupported("randint", low)
    if high_i is None:
        raise symbolic_helper._onnx_unsupported("randint", high)

    shape = symbolic_helper._maybe_get_const(shapes, "is")
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        randn = g.op(
            "RandomUniformLike",
            shape_const,
            low_f=low_i,
            high_f=high_i,
        )
    else:
        randn = g.op(
            "RandomUniform",
            shape_i=shape,
            low_f=low_i,
            high_f=high_i,
        )

    # cast to integer type
    int_dtype = _type_utils.JitScalarType.INT64
    randint = g.op("Cast", randn, to_i=int_dtype.onnx_type())
    if int_dtype != scalar_type:
        randint = g.op("Cast", randint, to_i=scalar_type.onnx_type())
    return randint


@_onnx_symbolic("aten::randint_like")
@_beartype.beartype
def randint_like(g: jit_utils.GraphContext, self, low, high, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    low_i = symbolic_helper._get_const(low, "i", "low")
    high_i = symbolic_helper._get_const(high, "i", "high")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.INT64
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    if low_i is None:
        raise symbolic_helper._onnx_unsupported("randint", low)
    if high_i is None:
        raise symbolic_helper._onnx_unsupported("randint", high)

    randn = g.op(
        "RandomUniformLike",
        self,
        low_f=low_i,
        high_f=high_i,
    )

    # cast to integer type
    int_dtype = _type_utils.JitScalarType.INT64
    randint = g.op("Cast", randn, to_i=int_dtype.onnx_type())
    if int_dtype != scalar_type:
        randint = g.op("Cast", randint, to_i=scalar_type.onnx_type())
    return randint


@_onnx_symbolic("aten::randn")
@_beartype.beartype
def randn(g: jit_utils.GraphContext, shapes, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        return g.op(
            "RandomNormalLike",
            shape_const,
            dtype_i=scalar_type.onnx_type(),
        )
    return g.op(
        "RandomNormal",
        shape_i=shape,
        dtype_i=scalar_type.onnx_type(),
    )


@_onnx_symbolic("aten::rand")
@_beartype.beartype
def rand(g: jit_utils.GraphContext, shapes, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        return g.op(
            "RandomUniformLike",
            shape_const,
            dtype_i=scalar_type.onnx_type(),
        )
    return g.op(
        "RandomUniform",
        shape_i=shape,
        dtype_i=scalar_type.onnx_type(),
    )


@_onnx_symbolic("aten::randn_like")
@_beartype.beartype
def randn_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.from_value(
            self, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    return g.op("RandomNormalLike", self, dtype_i=scalar_type.onnx_type())


@_onnx_symbolic("aten::rand_like")
@_beartype.beartype
def rand_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = _type_utils.JitScalarType.from_value(
            self, _type_utils.JitScalarType.FLOAT
        )
    return g.op(
        "RandomUniformLike", self, dtype_i=_type_utils.JitScalarType(dtype).onnx_type()
    )


@_onnx_symbolic("aten::rrelu")
@symbolic_helper.parse_args("v", "f", "f", "i", "none")
@_beartype.beartype
def rrelu(g: jit_utils.GraphContext, input, lower, upper, training, generator):
    if not training:
        slope = (upper + lower) / 2.0
        return g.op("LeakyRelu", input, alpha_f=slope)
    p = g.op("RandomUniformLike", input, high_f=upper, low_f=lower)
    return g.op("PRelu", input, p)


@_onnx_symbolic("aten::bernoulli")
@_beartype.beartype
def bernoulli(g: jit_utils.GraphContext, input, p=None, generator=None, out=None):
    if out is not None and not symbolic_helper._is_none(out):
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )

    dtype = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.UNDEFINED
    )
    if dtype == _type_utils.JitScalarType.UNDEFINED:
        return symbolic_helper._unimplemented(
            "Bernoulli", "input dtype not accessible", input
        )

    rands = g.op(
        "RandomUniformLike",
        input,
        high_f=1.0,
        low_f=0.0,
        dtype_i=dtype.onnx_type(),
    )
    prob = p if p is not None and not symbolic_helper._is_none(p) else input
    output = g.op("Less", rands, prob)
    return g.op("Cast", output, to_i=dtype.onnx_type())


@_onnx_symbolic("aten::log_sigmoid")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def log_sigmoid(g: jit_utils.GraphContext, input):
    p = g.op("Sigmoid", input)
    return g.op("Log", p)


@_onnx_symbolic("aten::erf")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def erf(g: jit_utils.GraphContext, input):
    return g.op("Erf", input)


@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    dim = symbolic_helper._get_tensor_rank(input)
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
            input,
        )

    if dim == 0:
        return symbolic_helper._reshape_helper(g, input, [1])
    if dim == 1:
        return g.op("Identity", input)
    # TODO: remove this as onnx opset 11 spec allows negative axes
    if end_dim < 0:
        end_dim = dim + end_dim
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1 and end_dim == dim - 1:
        return g.op("Flatten", input, axis_i=start_dim)
    if start_dim == 0 and end_dim == dim - 2:
        return g.op("Flatten", input, axis_i=end_dim + 1)

    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)


@_onnx_symbolic("aten::nonzero")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def nonzero(g: jit_utils.GraphContext, input):
    """Emitted from `torch.nonzero(x, as_tuple=False)`"""
    return t(g, g.op("NonZero", input))


@_onnx_symbolic("aten::nonzero_numpy")
# Emitted from `torch.nonzero(x, as_tuple=True)`
@_beartype.beartype
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    return unbind(g, nonzero(g, input), 1, _outputs=_outputs)


@_onnx_symbolic("aten::isnan")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def isnan(g: jit_utils.GraphContext, input):
    output = g.op("IsNaN", input)
    return output


@_onnx_symbolic("aten::any")
@_beartype.beartype
def _any(g: jit_utils.GraphContext, *args):
    # aten::any(Tensor self)
    if len(args) == 1:
        input = args[0]
        dim, keepdim = None, 0
    # aten::any(Tensor self, int[]? dim, bool keepdim)
    else:
        input, dim, keepdim = args
        # Can be int list or single int
        dim = symbolic_helper._parse_arg(dim, "t")
        dim = [int(d) for d in dim.view(-1)]
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
    input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT64)
    input_sum = symbolic_helper._reducesum_helper(
        g, input, axes_i=dim, keepdims_i=keepdim
    )
    return gt(g, input_sum, g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)))


@_onnx_symbolic("aten::all")
@_beartype.beartype
def _all(g: jit_utils.GraphContext, *args):
    input = g.op("Not", args[0])
    # aten::all(Tensor self)
    if len(args) == 1:
        return g.op("Not", _any(g, input))
    # aten::all(Tensor self, int[]? dim, bool keepdim)
    else:
        return g.op("Not", _any(g, input, args[1], args[2]))


@_onnx_symbolic("aten::narrow")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    return symbolic_helper._slice_helper(
        g, input, axes=[dim], starts=[start], ends=[start + length]
    )


@_onnx_symbolic("aten::argmax")
@symbolic_helper.parse_args("v", "v", "b")
@_beartype.beartype
def argmax(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMax")


@_onnx_symbolic("aten::argmin")
@symbolic_helper.parse_args("v", "v", "b")
@_beartype.beartype
def argmin(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMin")


@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    src_type = _type_utils.JitScalarType.from_value(
        src, _type_utils.JitScalarType.UNDEFINED
    )
    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("Scatter", self, index, src, axis_i=dim)
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        self_scalar_type = _type_utils.JitScalarType.from_value(self)
        if self_scalar_type != src_type:
            src = g.op("Cast", src, to_i=self_scalar_type.onnx_type())
        return g.op("Scatter", self, index, expand_as(g, src, index), axis_i=dim)


@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src):
    scalar_type = symbolic_helper._try_get_scalar_type(self)
    if scalar_type is None:
        return symbolic_helper._unimplemented(
            "scatter_add", "input dtype not accessible", self
        )
    sizes = symbolic_helper._get_tensor_sizes(self, allow_nonstatic=False)
    if sizes:
        to_add = g.op("Constant", value_t=torch.zeros(sizes, dtype=scalar_type.dtype()))
    else:
        to_add = zeros_like(g, self, scalar_type)
    to_add = symbolic_helper._scatter_helper(g, to_add, dim, index, src)
    return add(g, self, to_add)


@_onnx_symbolic("aten::log2")
@_beartype.beartype
def log2(g: jit_utils.GraphContext, self):
    _ln2 = 0.693147180559945309
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor(_ln2)))


@_onnx_symbolic("aten::is_floating_point")
@_beartype.beartype
def is_floating_point(g: jit_utils.GraphContext, self):
    if symbolic_helper._is_fp(self):
        return g.op("Constant", value_t=torch.BoolTensor([1]))
    return g.op("Constant", value_t=torch.BoolTensor([0]))


@_onnx_symbolic("aten::__is_")
@_beartype.beartype
def __is_(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_none(other):
        if symbolic_helper._is_none(self):
            return g.op("Constant", value_t=torch.BoolTensor([1]))
        return g.op("Constant", value_t=torch.BoolTensor([0]))
    return eq(g, self, other)


@_onnx_symbolic("aten::__isnot_")
@wrap_logical_op_with_negation
@_beartype.beartype
def __isnot_(g: jit_utils.GraphContext, self, other):
    return __is_(g, self, other)


@_onnx_symbolic("aten::one_hot")
@_beartype.beartype
def one_hot(g: jit_utils.GraphContext, self, num_classes):
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    # onnxruntime supports limited type combinations for OneHot.
    if _type_utils.JitScalarType.from_value(
        num_classes, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
        _type_utils.JitScalarType.INT,
        _type_utils.JitScalarType.INT16,
    }:
        num_classes = g.op("Cast", num_classes, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op("OneHot", self, num_classes, values, axis_i=-1)


@_onnx_symbolic("aten::gather")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True", self)
    # NOTE: This workaround is needed since GatherElement is only supported
    #       since opset 11, and Gather in ONNX is not the same as torch.gather.
    scalar_type = _type_utils.JitScalarType.from_value(self)
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    depth = size(g, self, g.op("Constant", value_t=torch.LongTensor([dim])))
    index = g.op(
        "Cast",
        g.op("OneHot", index, depth, values, axis_i=dim),
        to_i=scalar_type.onnx_type(),
    )
    mul = g.op("Mul", symbolic_helper._unsqueeze_helper(g, self, [dim + 1]), index)
    return symbolic_helper._reducesum_helper(g, mul, axes_i=[dim], keepdims_i=0)


@symbolic_helper.parse_args("v", "is", "i", "i")
@_beartype.beartype
def _var_mean(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    return symbolic_helper._var_mean_helper(g, input, dim, correction, keepdim)


@_onnx_symbolic("aten::std")
@_beartype.beartype
def std(g: jit_utils.GraphContext, input, *args):
    var, _ = var_mean(g, input, *args)
    return g.op("Sqrt", var)


@_onnx_symbolic("aten::var")
@_beartype.beartype
def var(g: jit_utils.GraphContext, input, *args):
    var, _ = var_mean(g, input, *args)
    return var


@_onnx_symbolic("aten::var_mean")
@_beartype.beartype
def var_mean(g: jit_utils.GraphContext, input, *args):
    if len(args) == 1:
        return _var_mean(g, input, None, args[0], None)
    else:
        return _var_mean(g, input, *args)


@_onnx_symbolic("aten::std_mean")
@_beartype.beartype
def std_mean(g: jit_utils.GraphContext, input, *args):
    var, mean = var_mean(g, input, *args)
    return g.op("Sqrt", var), mean


@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    return g.op("ReduceLogSumExp", input, axes_i=dim, keepdims_i=keepdim)


@_onnx_symbolic("aten::arange")
@_beartype.beartype
def arange(g: jit_utils.GraphContext, *args):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("arange", *args)

    @_beartype.beartype
    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    @_beartype.beartype
    def _float_step_convert(range_tensor):
        if symbolic_helper._is_fp(range_tensor):
            range_tensor = g.op(
                "Cast",
                g.op("Ceil", range_tensor),
                to_i=_type_utils.JitScalarType.INT64.onnx_type(),
            )
        return range_tensor

    if len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[1])
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, end=args[0], dtype=dtype
        )
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        range_tensor = _float_step_convert(end)
        arange_tensor = symbolic_helper._squeeze_helper(
            g, nonzero(g, ones(g, range_tensor, dtype, None, None)), [1]
        )
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[3])
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], step=args[2], dtype=dtype
        )
        step = symbolic_helper._unsqueeze_helper(g, step, [0])
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        start = symbolic_helper._unsqueeze_helper(g, start, [0])
        range_tensor = _float_step_convert(g.op("Div", g.op("Sub", end, start), step))
        arange_tensor = symbolic_helper._squeeze_helper(
            g, nonzero(g, ones(g, range_tensor, None, None, None)), [1]
        )
        arange_tensor = g.op("Add", g.op("Mul", arange_tensor, step), start)
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], dtype=dtype
        )
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        start = symbolic_helper._unsqueeze_helper(g, start, [0])
        range_tensor = _float_step_convert(g.op("Sub", end, start))
        arange_tensor = g.op(
            "Add",
            symbolic_helper._squeeze_helper(
                g, nonzero(g, ones(g, range_tensor, dtype, *(args[3:]))), [1]
            ),
            start,
        )
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )

    return symbolic_helper._unimplemented("aten::arange", f"with {len(args)} arguments")


@_onnx_symbolic("aten::linspace")
@_beartype.beartype
def linspace(
    g: jit_utils.GraphContext, start, end, steps, dtype, layout, device, pin_memory
):
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    step = div(
        g,
        sub(g, end, start),
        sub(g, steps, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))),
    )
    return add(g, mul(g, range_tensor, step), start)


@_onnx_symbolic("aten::lift")
@_beartype.beartype
def lift(g: jit_utils.GraphContext, self):
    # at::lift() is a no-op from the perspective of tracing for onnx
    return self


@_onnx_symbolic("aten::masked_fill")
@_beartype.beartype
def masked_fill(g: jit_utils.GraphContext, self, mask, value):
    """Implement the masked_fill functionality available for a pytorch tensor in ONNX.

    Fills elements of the input tensor with `value` where `mask` is True.
    """
    mask = g.op("Cast", mask, to_i=_C_onnx.TensorProtoDataType.BOOL)
    value = symbolic_helper._maybe_get_scalar(value)
    return g.op("Where", mask, symbolic_helper._if_scalar_type_as(value, self), self)


@_onnx_symbolic("aten::masked_fill_")
@_beartype.beartype
def masked_fill_(g: jit_utils.GraphContext, self, mask, value):
    return masked_fill(g, self, mask, value)


@_onnx_symbolic("aten::index")
@_beartype.beartype
def index(g: jit_utils.GraphContext, self, index):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index", self, index, overload_name="Tensor")

    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    @_beartype.beartype
    def try_mask_to_index(index):
        if not symbolic_helper._is_none(index) and (
            _type_utils.JitScalarType.from_value(
                index, _type_utils.JitScalarType.UNDEFINED
            )
            == _type_utils.JitScalarType.UINT8
            or symbolic_helper._is_bool(index)
        ):
            if g.opset < 9:
                raise errors.SymbolicValueError(
                    "Exporting masked indices are only supported after ONNX opset 9.",
                    self,
                )
            warnings.warn(
                "Exporting aten::index operator with indices of type Byte. "
                "Only 1-D indices are supported. In any other case, "
                "this will produce an incorrect ONNX graph."
            )
            index = symbolic_helper._squeeze_helper(g, nonzero(g, index), [1])
        return index

    indices = [try_mask_to_index(idx) for idx in indices]
    if len(indices) == 1:
        return symbolic_helper._select_helper(
            g, self, 0, indices[0], apply_reshape=False
        )
    else:
        # Multiple tensors as indices. Each tensor could either be
        #   1. prim::Constant()
        #           representing ":" in python indexing. E.g. tensor[:, :]
        #   2. prim::Constant[value=...] or tensor output
        #           representing advanced indexing. E.g. tensor[[0, 1], [2, 0]].
        # For more info on advanced indexing,
        # check https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

        # Consider a general case of
        #       t: [x_1, y_1, y_2, ..., x_m, ..., y_n]
        # where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes for ":".
        # Same results can be achieved through transposing t into
        #       t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n]
        # and use gatherND. However ONNX does not have gatherND, to use 1d gather we'll need to flatten t
        # and process the tensor indices.
        #       t: [x_1 * x_2 * ... * x_m, y_1 * y_2 * ... * y_n]
        #       tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j))
        # After gather, reshape and transpose back.
        adv_idx_indices = [
            i for i, idx in enumerate(indices) if not symbolic_helper._is_none(idx)
        ]

        if len(adv_idx_indices) == 0:
            return self
        elif len(adv_idx_indices) == 1:
            return index_select(
                g, self, adv_idx_indices[0], indices[adv_idx_indices[0]]
            )
        else:
            rank = symbolic_helper._get_tensor_rank(self)
            if rank is None:
                return symbolic_helper._unimplemented(
                    "aten::index",
                    "operator of advanced indexing on tensor of unknown rank. "
                    "Try turning on shape inference during export: "
                    "torch.onnx._export(..., onnx_shape_inference=True).",
                    self,
                )
            # TODO: If indexing is supported natively in ONNX in future opsets,
            #       update the warning to recommend exporting with higher opset version.
            warnings.warn(
                "Exporting aten::index operator of advanced indexing in opset "
                f"{GLOBALS.export_onnx_opset_version}"
                " is achieved by combination of multiple ONNX operators, "
                "including Reshape, Transpose, Concat, and Gather. "
                "If indices include negative values, the exported graph will produce incorrect results."
            )
            adv_idx_count = len(adv_idx_indices)
            shape_tensor = _shape_as_tensor(g, self)
            dim_tensor_list = [
                g.op(
                    "Gather",
                    shape_tensor,
                    g.op("Constant", value_t=torch.LongTensor([dim])),
                    axis_i=0,
                )
                for dim in range(rank)
            ]

            self = g.op(
                "Transpose",
                self,
                perm_i=adv_idx_indices
                + [i for i in range(rank) if i not in adv_idx_indices],
            )
            self = g.op("Flatten", self, axis_i=adv_idx_count)

            # Note that tensor indices will be broadcasted while accumulating. Thus we get the final subarray shape as well.
            cum_adv_index = indices[adv_idx_indices[-1]]
            multiplier = dim_tensor_list[adv_idx_indices[-1]]
            for i in range(adv_idx_count - 2, -1, -1):
                adv_index = g.op("Mul", indices[adv_idx_indices[i]], multiplier)
                cum_adv_index = g.op("Add", cum_adv_index, adv_index)
                multiplier = g.op(
                    "Mul", multiplier, dim_tensor_list[adv_idx_indices[i]]
                )

            # perform gather
            self = index_select(g, self, 0, cum_adv_index)

            cum_adv_index_shape_tensor = _shape_as_tensor(g, cum_adv_index)
            # check if all advanced indices are consecutive.
            # Refer to https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
            # to understand how the subarray position is decided.
            if adv_idx_indices == list(
                range(adv_idx_indices[0], adv_idx_indices[-1] + 1)
            ):
                # unfold regular index axes
                folded_adv_idx_shape_list = [
                    g.op("Constant", value_t=torch.LongTensor([-1]))
                ] + [
                    dim_tensor_list[i] for i in range(rank) if i not in adv_idx_indices
                ]
                folded_adv_idx_shape = g.op(
                    "Concat", *folded_adv_idx_shape_list, axis_i=0
                )
                self = symbolic_helper._reshape_helper(g, self, folded_adv_idx_shape)

                # Transpose folded advanced indexed axis to its original location.
                adv_idx_permute = (
                    list(range(1, adv_idx_indices[0] + 1))
                    + [0]
                    + list(range(adv_idx_indices[0] + 1, rank - adv_idx_count + 1))
                )
                self = g.op("Transpose", self, perm_i=adv_idx_permute)

                # unfold advanced index axes
                final_shape_list = (
                    [dim_tensor_list[i] for i in range(adv_idx_indices[0])]
                    + [cum_adv_index_shape_tensor]
                    + [
                        dim_tensor_list[i]
                        for i in range(adv_idx_indices[0], rank)
                        if i not in adv_idx_indices
                    ]
                )
                final_shape = g.op("Concat", *final_shape_list, axis_i=0)
            else:
                final_shape = g.op(
                    "Concat",
                    cum_adv_index_shape_tensor,
                    *[
                        dim_tensor_list[i]
                        for i in range(rank)
                        if i not in adv_idx_indices
                    ],
                    axis_i=0,
                )

            return symbolic_helper._reshape_helper(g, self, final_shape)


@_onnx_symbolic("aten::linalg_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def linalg_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
    ord_value = None
    if dim is None:
        if symbolic_helper._is_none(ord):
            self = symbolic_helper._reshape_helper(g, self, [-1])
            ord = g.op("Constant", value_t=torch.LongTensor([2]))
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "dim", "Input rank must be known at export time.", self
            )
        if self_dim == 1:
            ord_value = symbolic_helper._parse_arg(ord, "f")
        else:
            dim = [0, 1]
    else:
        if len(dim) == 1:
            if symbolic_helper._is_none(ord):
                ord = g.op("Constant", value_t=torch.LongTensor([2]))
            ord_value = symbolic_helper._parse_arg(ord, "f")
    if ord_value:
        return linalg_vector_norm(g, self, ord_value, dim, keepdim, dtype)
    return linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
@_beartype.beartype
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: List[int],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html
    ord_value = symbolic_helper._parse_arg(ord, "s")
    if ord_value == "fro":
        return frobenius_norm(g, self, dim, keepdim)
    elif ord_value == "nuc":
        return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==nuc", self)
    else:
        ord_value = symbolic_helper._parse_arg(ord, "f")
        if ord_value is None:
            return frobenius_norm(g, self, dim, keepdim)
        if ord_value == 2 or ord_value == -2:
            # ord = 2/-2 unimplemented due to lack of operators
            # used to calculate singular values
            return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==2", self)
        # Wrap the dim vector to handle negative dim values
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "linalg.matrix_norm", "Input rank must be known at export time.", self
            )
        # Common implementation for cases with
        # ord = 1/-1 and ord = inf/-inf
        if dim[0] < 0:
            dim[0] += self_dim
        if dim[1] < 0:
            dim[1] += self_dim

        if ord_value == math.inf or ord_value == -math.inf:
            dim[0], dim[1] = dim[1], dim[0]
        if dim[1] > dim[0] and not keepdim:
            dim[1] -= 1
        sum = symbolic_helper._reducesum_helper(
            g, g.op("Abs", self), axes_i=[dim[0]], keepdims_i=keepdim
        )
        if ord_value > 0:
            result, indices = max(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        else:
            result, indices = min(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        return result


@_onnx_symbolic("aten::linalg_cross")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def linalg_cross(g: jit_utils.GraphContext, input, other, dim=-1):
    return cross(g, input, other, dim)


@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "is", "b")
@_beartype.beartype
def frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    sqr = g.op("Mul", self, self)
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, axes_i=dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)


@_onnx_symbolic("aten::multinomial")
@symbolic_helper.parse_args("v", "i", "b", "v")
@_beartype.beartype
def multinomial(
    g: jit_utils.GraphContext, input, num_samples, replacement=False, generator=None
):
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Multinomial", "generator is not supported for multinomial", input
        )
    if not replacement and num_samples > 1:
        symbolic_helper._unimplemented(
            "Multinomial",
            "replacement=False when num_samples > 1 is not supported for multinomial",
            input,
        )

    log_input = log(g, input)
    return g.op(
        "Multinomial",
        log_input,
        dtype_i=_C_onnx.TensorProtoDataType.INT64,
        sample_size_i=num_samples,
    )


@_onnx_symbolic("aten::baddbmm")
@_beartype.beartype
def baddbmm(g: jit_utils.GraphContext, self, batch1, batch2, beta, alpha):
    scalar_type = _type_utils.JitScalarType.from_value(self)
    batch_mul = matmul(g, batch1, batch2)
    mul_a = mul(
        g,
        batch_mul,
        g.op("Cast", alpha, to_i=scalar_type.onnx_type()),
    )
    mul_b = mul(
        g,
        self,
        g.op("Cast", beta, to_i=scalar_type.onnx_type()),
    )
    return add(g, mul_a, mul_b)


@_onnx_symbolic("aten::meshgrid")
@symbolic_helper.parse_args("v", "s")
@_beartype.beartype
def meshgrid(g: jit_utils.GraphContext, tensor_list, indexing: Optional[str] = None):
    if indexing is None:
        indexing = "ij"
    elif indexing not in {"ij", "xy"}:
        raise errors.SymbolicValueError(
            f"Unsupported indexing: {indexing}", tensor_list
        )
    unpacked_tensor_list = symbolic_helper._unpack_list(tensor_list)
    if indexing == "xy":
        unpacked_tensor_list[:2] = unpacked_tensor_list[1::-1]
    tensors = [
        symbolic_helper._reshape_helper(
            g, t, g.op("Constant", value_t=torch.LongTensor([-1]))
        )
        for t in unpacked_tensor_list
    ]
    tensors_shape = [g.op("Shape", t) for t in tensors]
    out_shape = g.op("Concat", *tensors_shape, axis_i=0)
    out = []
    for i, t in enumerate(tensors):
        shape_i = [g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))] * len(
            tensors
        )
        shape_i[i] = tensors_shape[i]
        t_reshaped = _reshape_from_tensor(g, t, g.op("Concat", *shape_i, axis_i=0))
        out.append(g.op("Expand", t_reshaped, out_shape))
    if indexing == "xy":
        out[0], out[1] = out[1], out[0]
    return g.op("prim::ListConstruct", *out)


@_onnx_symbolic("aten::remainder")
@_beartype.beartype
def remainder(g: jit_utils.GraphContext, input, other):
    div = _floor_divide(g, input, other)
    quo = g.op("Mul", div, other)
    return g.op("Sub", input, quo)


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
@_beartype.beartype
def gelu(g: jit_utils.GraphContext, self: torch._C.Value, approximate: str = "none"):
    if approximate == "tanh":
        kBeta = math.sqrt(2 / math.pi)
        kKappa = 0.044715

        beta = torch.tensor(kBeta, dtype=torch.double)
        kappa = torch.tensor(kKappa, dtype=torch.double)
        one = torch.tensor(1.0, dtype=torch.double)
        half = torch.tensor(0.5, dtype=torch.double)

        self_cube = mul(g, self, mul(g, self, self))
        inner = mul(g, beta, add(g, self, mul(g, kappa, self_cube)))
        return mul(g, half, mul(g, self, add(g, one, g.op("Tanh", inner))))
    else:
        _sqrt2 = 1.4142135623730951
        erf = g.op("Erf", g.op("Div", self, torch.tensor(_sqrt2, dtype=torch.double)))
        erf_plusone = add(
            g, erf, g.op("Constant", value_t=torch.tensor(1, dtype=torch.double))
        )
        return mul(
            g,
            mul(g, self, erf_plusone),
            g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.double)),
        )


@_onnx_symbolic("aten::group_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "i", "v", "v", "f", "i")
@_beartype.beartype
def group_norm(
    g: jit_utils.GraphContext, input, num_groups, weight, bias, eps, cudnn_enabled
):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "group_norm",
            input,
            weight,
            bias,
            num_groups_i=num_groups,
            eps_f=eps,
            cudnn_enabled_i=cudnn_enabled,
        )

    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if channel_size is not None:
        assert channel_size % num_groups == 0
    input_rank = symbolic_helper._get_tensor_rank(input)
    if input_rank is None:
        return symbolic_helper._unimplemented("group_norm", "unknown input rank", input)
    # 0 in the shape list keeps dimension value unchanged.
    shape = [0, num_groups, -1]
    input_reshaped = symbolic_helper._reshape_helper(
        g, input, g.op("Constant", value_t=torch.LongTensor(shape))
    )

    # C is always divisible by num_groups
    # Due to shape difference. we need to apply weight and bias after
    # instance norm computation and reshape
    weight_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [1.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        ),
    )
    bias_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [0.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        ),
    )

    norm_reshaped = g.op(
        "InstanceNormalization", input_reshaped, weight_, bias_, epsilon_f=eps
    )
    norm = symbolic_helper._reshape_helper(g, norm_reshaped, g.op("Shape", input))

    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor(
            [1.0], dtype=_type_utils.JitScalarType.from_value(input).dtype()
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().mustBeNone():
        bias_value = torch.tensor(
            [0.0], dtype=_type_utils.JitScalarType.from_value(input).dtype()
        )
        bias = g.op("Constant", value_t=bias_value)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    axes = list(range(1, input_rank - 1))
    return add(
        g,
        mul(g, norm, symbolic_helper._unsqueeze_helper(g, weight, axes)),
        symbolic_helper._unsqueeze_helper(g, bias, axes),
    )


@_onnx_symbolic("aten::_weight_norm")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def _weight_norm(g: jit_utils.GraphContext, weight_v, weight_g, dim):
    rank = symbolic_helper._get_tensor_rank(weight_v)
    if rank is not None:
        # W = g * ((v) / ||v||)
        # Compute norm_except_dim for l2 norm. dim = None means over all dims
        # torch's weight_norm module sets dim = -1 if it's None.
        # This conflicts the logic for negative axes to access dims backwards
        # TODO: Might need a fix in torch group_norm module
        axes = list(range(rank))
        if dim is not None:
            if dim < -1:
                dim += rank
            if dim != -1:
                axes.remove(dim)
        norm_v = norm(g, weight_v, 2, axes, 1)
        div = g.op("Div", weight_v, norm_v)
        return g.op("Mul", div, weight_g)
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("_weight_norm", weight_v, weight_g, dim_i=dim)

    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of _weight_norm for tensor of unknown rank.",
        weight_v,
    )


@_onnx_symbolic("aten::dim")
@_beartype.beartype
def dim(g: jit_utils.GraphContext, self):
    """Implement the dim functionality available for a pytorch tensor in ONNX"""
    # ONNX does not support dim directly in this opset so we can use 2 ops to get the info
    shape = g.op("Shape", self)
    return g.op("Size", shape)


@_onnx_symbolic("aten::__contains_")
@_beartype.beartype
def __contains_(g: jit_utils.GraphContext, self, element):
    unpacked_list = symbolic_helper._unpack_list(self)
    if all(
        symbolic_helper._is_constant(x) for x in unpacked_list
    ) and symbolic_helper._is_constant(element):
        return g.op(
            "Constant",
            value_t=torch.tensor(
                symbolic_helper._node_get(element.node(), "value")
                in (symbolic_helper._node_get(x.node(), "value") for x in unpacked_list)
            ),
        )

    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of __contains__ for non-constant list or element.",
        self,
    )


@_onnx_symbolic("aten::__getitem_")
@_beartype.beartype
def __getitem_(g: jit_utils.GraphContext, self, i):
    return select(g, self, g.op("Constant", value_t=torch.tensor([0])), i)


@_onnx_symbolic("aten::item")
@_beartype.beartype
def item(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("aten::take")
@_beartype.beartype
def take(g: jit_utils.GraphContext, self, index):
    self_flattened = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    out = index_select(g, self_flattened, 0, index)
    out = reshape_as(g, out, index)
    return out


@_beartype.beartype
def _kl_div_log_target_impl(g: jit_utils.GraphContext, input, target):
    diff_ = sub(g, target, input)
    exp_ = exp(g, target)
    output = mul(g, exp_, diff_)
    return output


@_beartype.beartype
def _kl_div_non_log_target_impl(g: jit_utils.GraphContext, input, target):
    log_ = log(g, target)
    diff_ = sub(g, log_, input)
    output_pos = mul(g, target, diff_)
    zeros_ = zeros_like(g, output_pos)
    mask_ = gt(g, target, g.op("Constant", value_t=torch.tensor(0)))
    output = where(g, mask_, output_pos, zeros_)
    return output


@_onnx_symbolic("aten::kl_div")
@symbolic_helper.parse_args("v", "v", "i", "b")
@_beartype.beartype
def kl_div(g: jit_utils.GraphContext, input, target, reduction, log_target):
    if log_target:
        output = _kl_div_log_target_impl(g, input, target)
    else:
        output = _kl_div_non_log_target_impl(g, input, target)

    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        return symbolic_helper._reducesum_helper(g, output, keepdims_i=0)
    else:
        return symbolic_helper._onnx_unsupported(
            "kl_div with reduction other than none, mean, or sum.", input
        )


@_onnx_symbolic("aten::mse_loss")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def mse_loss(g: jit_utils.GraphContext, input, target, reduction):
    output = mul(g, sub(g, input, target), sub(g, input, target))
    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        return symbolic_helper._reducesum_helper(g, output, keepdims_i=0)
    else:
        return symbolic_helper._onnx_unsupported(
            "mse_loss with reduction other than none, mean, or sum.", input
        )


@_onnx_symbolic("aten::as_strided")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "is", "i")
@_beartype.beartype
def as_strided(g: jit_utils.GraphContext, self, sizes, strides, offset=None):
    sizes = symbolic_helper._maybe_get_const(sizes, "is")
    rank = len(strides)
    self_1d = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    ind: Optional[torch.Tensor]
    if not symbolic_helper._is_value(sizes):
        ind = torch.tensor([0], dtype=torch.long)
        for i, (size, stride) in enumerate(zip(sizes, strides)):
            r_size = [1] * rank
            r_size[i] = -1
            ind = ind + torch.arange(size).view(r_size) * stride
        if offset:
            ind = ind + offset
        return g.op("Gather", self_1d, g.op("Constant", value_t=ind))
    else:
        ind = None
        for i, stride in enumerate(strides):
            r_size = [1] * rank
            r_size[i] = -1
            size = select(
                g,
                sizes,
                g.op("Constant", value_t=torch.tensor([0])),
                g.op("Constant", value_t=torch.tensor(i)),
            )
            tmp_ind = symbolic_helper._reshape_helper(
                g,
                arange(g, size, 4, None, None, None),
                g.op("Constant", value_t=torch.tensor(r_size)),
            )
            tmp_ind = g.op(
                "Mul", tmp_ind, g.op("Constant", value_t=torch.tensor([stride]))
            )
            if ind is None:
                ind = tmp_ind
            else:
                ind = g.op("Add", ind, tmp_ind)
        if offset:
            ind = g.op("Add", ind, g.op("Constant", torch.tensor([offset])))
        return g.op("Gather", self_1d, ind)


@_onnx_symbolic("aten::__derive_index")
@_beartype.beartype
def __derive_index(g: jit_utils.GraphContext, index, start, step):
    return g.op("Add", start, g.op("Mul", index, step))


@_onnx_symbolic("aten::__range_length")
# Source code for aten op can be found here: pytorch/torch/csrc/jit/runtime/register_prim_ops.cpp
# if (step > 0 && lo < hi) {
#   push(stack, 1 + (hi - 1 - lo) / step);
# } else if (step < 0 && lo > hi) {
#   push(stack, 1 + (lo - 1 - hi) / (0 - step));
# } else {
#  push(stack, 0);
# }
@_beartype.beartype
def __range_length(g: jit_utils.GraphContext, lo, hi, step):
    sub = g.op("Sub", hi, lo)
    div = g.op("Ceil", true_divide(g, sub, step))
    return g.op("Cast", div, to_i=_C_onnx.TensorProtoDataType.INT64)


@_onnx_symbolic("aten::linear")
@_beartype.beartype
def linear(g: jit_utils.GraphContext, input, weight, bias):
    rank = symbolic_helper._get_tensor_rank(input)
    weight = t(g, weight)
    if rank == 2 and not bias.node().mustBeNone():
        alpha = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        beta = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        output = addmm(g, bias, input, weight, alpha, beta)
    else:
        output = matmul(g, input, weight)
        if not bias.node().mustBeNone():
            output = add(g, bias, output)

    return output


@_onnx_symbolic("aten::hann_window")
@symbolic_helper.parse_args("v", "b", "i", "v", "v", "v", "v")
@_beartype.beartype
def hann_window(
    g: jit_utils.GraphContext,
    window_length,
    periodic=True,
    dtype: Optional[int] = None,
    layout=None,
    device=None,
    pin_memory=None,
    requires_grad=False,
):
    if dtype is None:
        dtype_ = torch.get_default_dtype()
        if not dtype_ or not dtype_.is_floating_point:
            dtype_ = torch.float
        scalar_type = _type_utils.JitScalarType.from_dtype(dtype_)
    else:
        scalar_type = _type_utils.JitScalarType(dtype)

    n_array = arange(g, window_length, 4, None, None, None)
    output = g.op("Cast", n_array, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    output = mul(
        g, g.op("Constant", value_t=torch.tensor(math.pi, dtype=torch.float)), output
    )

    if periodic is False:
        window_length = sub(
            g, window_length, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int))
        )
    output = div(g, output, window_length)
    output = g.op(
        "Cast",
        square(g, sin(g, output)),
        to_i=scalar_type.onnx_type(),
    )

    return output


@_onnx_symbolic("aten::mv")
@_beartype.beartype
def mv(g: jit_utils.GraphContext, self, vec):
    return matmul(g, self, vec)


@_onnx_symbolic("aten::dot")
@_beartype.beartype
def dot(g: jit_utils.GraphContext, self, other):
    return matmul(g, self, other)


@_onnx_symbolic("aten::movedim")
@symbolic_helper.parse_args("v", "t", "t")
@_beartype.beartype
def movedim(g: jit_utils.GraphContext, self, source, destination):
    # This is a pythonic implementation mostly taken from aten/src/ATen/native/TensorShape.cpp::movedim
    source = source.view(-1)
    destination = destination.view(-1)

    assert source.size() == destination.size()

    if (source == destination).all():
        return self

    self_rank = symbolic_helper._get_tensor_rank(self)
    assert self_rank is not None

    perm = list(range(self_rank))

    src_dims = perm.copy()
    dst_dims = perm.copy()

    for src, dst in zip(source.tolist(), destination.tolist()):
        perm[dst] = src
        src_dims[src] = -1
        dst_dims[dst] = -1

    src_dims = [dim for dim in src_dims if dim != -1]
    dst_dims = [dim for dim in dst_dims if dim != -1]

    for src, dst in zip(src_dims, dst_dims):
        perm[dst] = src

    return g.op("Transpose", self, perm_i=perm)


@_onnx_symbolic("aten::fill")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def fill(g: jit_utils.GraphContext, self, value):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    return full_like(g, self, value, scalar_type)


@_onnx_symbolic("aten::index_add")
@_beartype.beartype
def index_add(g: jit_utils.GraphContext, self, dim, index, other, alpha=None):
    warnings.warn(
        "Warning: ONNX export does not support duplicated values in 'index' field, "
        + "this will cause the ONNX model to be incorrect."
    )

    # ONNX does not support "alpha" argument, unlike aten index_add
    # See: https://github.com/pytorch/pytorch/pull/65993#issuecomment-953151102 for more context
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        return symbolic_helper._unimplemented("index_add", "alpha != 1", self)

    dim = symbolic_helper._maybe_get_const(dim, "i")
    if dim is None:
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting 'index_add_()' function with "
            "unknown 'dim' value.",
            self,
        )

    self_dim_rank = symbolic_helper._get_tensor_rank(self)
    other_dim_rank = symbolic_helper._get_tensor_rank(other)

    if self_dim_rank is None or other_dim_rank is None:
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting 'index_add_()' function while "
            "the rank of self tensor or tensor to be added is unknown.",
            self,
        )

    if other_dim_rank != self_dim_rank:
        delta = self_dim_rank - other_dim_rank
        for i in range(delta):
            other = symbolic_helper._unsqueeze_helper(
                g, other, [symbolic_helper._get_tensor_rank(other)]
            )

    other_dim_size = symbolic_helper._get_tensor_dim_size(other, dim)
    self_dim_size = symbolic_helper._get_tensor_dim_size(self, dim)

    if (other_dim_size is not None) and (self_dim_size is not None):
        if other_dim_size > self_dim_size:
            raise errors.SymbolicValueError(
                "ONNX export does not support exporting 'index_add_()' function with "
                "duplicated values in 'index' parameter yet.",
                self,
            )

    # Construct a new shape. It's almost as same as self except the size of the 'dim'
    # dimension is 1, so that we can expand other dimensions as expected.
    new_shape_axes = list(range(self_dim_rank))
    new_shape_starts = [0 for i in range(self_dim_rank)]
    new_shape_ends = [sys.maxsize if (i != dim) else 1 for i in range(self_dim_rank)]

    new_shape = symbolic_helper._slice_helper(
        g, self, axes=new_shape_axes, starts=new_shape_starts, ends=new_shape_ends
    )
    other = expand_as(g, other, new_shape)

    for i in range(dim):
        index = symbolic_helper._unsqueeze_helper(g, index, [0])

    for i in range(self_dim_rank - dim - 1):
        index = symbolic_helper._unsqueeze_helper(
            g, index, [symbolic_helper._get_tensor_rank(index)]
        )

    return scatter_add(g, self, dim, expand_as(g, index, other), other)


@_onnx_symbolic("aten::roll")
@symbolic_helper.parse_args("v", "is", "is")
@_beartype.beartype
def roll(g: jit_utils.GraphContext, self, shifts, dims):
    assert len(shifts) == len(dims)

    result = self
    for i in range(len(shifts)):
        shapes = []
        shape = symbolic_helper._slice_helper(
            g, result, axes=[dims[i]], starts=[-shifts[i]], ends=[sys.maxsize]
        )
        shapes.append(shape)
        shape = symbolic_helper._slice_helper(
            g, result, axes=[dims[i]], starts=[0], ends=[-shifts[i]]
        )
        shapes.append(shape)
        result = g.op("Concat", *shapes, axis_i=dims[i])

    return result


@_onnx_symbolic("aten::cross")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def cross(g: jit_utils.GraphContext, input, other, dim=None):
    dim = symbolic_helper._get_dim_for_cross(input, dim)
    # If we have two tensors such that
    # A = [a, b, c], B = [d, e, f], we permute the tensor such that we have
    # After first roll,
    # A' = [b, c, a], B' = [f, d, e], so that we calculate (b*f, c*d, a*e)
    roll_x_1 = roll(g, input, [2], [dim])
    roll_y_1 = roll(g, other, [1], [dim])
    # After second roll,
    # A' = [c, a, b], B' = [e, f, d], so that we calculate (c*e, a*f, b*d)
    roll_x_2 = roll(g, input, [1], [dim])
    roll_y_2 = roll(g, other, [2], [dim])
    # cross product is calculated as
    # result = [(b*f - c*e), (c*d - a*f), (a*e - b*d)]
    return sub(g, mul(g, roll_x_1, roll_y_1), mul(g, roll_x_2, roll_y_2))


@_onnx_symbolic("aten::cdist")
@_beartype.beartype
def cdist(
    g: jit_utils.GraphContext,
    x1,
    x2,
    p=2.0,
    compute_mode="use_mm_for_euclid_dist_if_necessary",
):
    # X1.shape = (B * P * D), X2.shape = (B * R * D)
    # In order to respect numpy style broadcasting as demonstrated in
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    # we unsqueeze both input tensors
    # Currently we ignore the 'compute_mode' variable as we use default to
    # using matrix multiplication to calculate the euclidean distance
    rank = symbolic_helper._get_tensor_rank(x1)
    assert rank is not None
    broadcasted_x1 = symbolic_helper._unsqueeze_helper(g, x1, [rank - 1])
    broadcasted_x2 = symbolic_helper._unsqueeze_helper(g, x2, [rank - 2])
    return pairwise_distance(
        g, broadcasted_x1, broadcasted_x2, p, eps=1e-06, keepdim=False
    )


@_onnx_symbolic("aten::lerp")
@_beartype.beartype
def lerp(g: jit_utils.GraphContext, self, end, weight):
    # Conditional for better numeric. This has been discussed in
    # https://github.com/pytorch/pytorch/pull/18871
    diff = g.op("Sub", end, self)
    return where(
        g,
        g.op("Less", weight, g.op("Constant", value_t=torch.tensor(0.5))),
        g.op("Add", self, g.op("Mul", weight, diff)),
        g.op(
            "Sub",
            end,
            g.op(
                "Mul",
                diff,
                g.op("Sub", g.op("Constant", value_t=torch.tensor(1.0)), weight),
            ),
        ),
    )


@_onnx_symbolic("aten::broadcast_tensors")
@_beartype.beartype
def broadcast_tensors(g: jit_utils.GraphContext, self):
    all_tensors = symbolic_helper._unpack_list(self)
    t_with_final_shape = zeros_like(g, all_tensors[0])

    # Add operator supports multidirectional broadcasting. So we leverage this function
    # to infer the final shape generated by the broadcast.
    for t in all_tensors:
        t_with_final_shape = add(g, t_with_final_shape, t)

    t_list = [expand_as(g, t, t_with_final_shape) for t in all_tensors]
    return g.op("prim::ListConstruct", *t_list)


@_onnx_symbolic("aten::is_pinned")
def is_pinned(g: jit_utils.GraphContext, self, device=None):
    # Unused by ONNX.
    return None


@_onnx_symbolic("prim::ConstantSplit")
@_beartype.beartype
def prim_constant_split(g: jit_utils.GraphContext, self, split_size, dim):
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented(
            "prim::ConstantSplit", "unknown dimension size", self
        )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=len(splits))


# TODO: It would be better to export this as a chunk directly, as this is
# less sensitive to changes in input size.
# TODO: Once we have proper scoping, stop reimplementing chunk, delete this
# method, and use the desugared version
@_onnx_symbolic("prim::ConstantChunk")
@_beartype.beartype
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(self, dim)
    if dim_size is None:
        return symbolic_helper._unimplemented(
            "prim::ConstantChunk", "unknown dimension size", self
        )
    split_size = (dim_size + chunks - 1) // chunks
    return prim_constant_split(g, self, split_size, dim)


@_onnx_symbolic("prim::shape")
@_beartype.beartype
def prim_shape(g: jit_utils.GraphContext, self):
    return g.op("Shape", self)


@_onnx_symbolic("prim::max")
@_beartype.beartype
def prim_max(g: jit_utils.GraphContext, self, other):
    return symbolic_helper._op_with_optional_float_cast(
        g, "Max", self, other, opset_before=12
    )


@_onnx_symbolic("prim::min")
@_beartype.beartype
def prim_min(g: jit_utils.GraphContext, self, other=None):
    if not other:
        if symbolic_helper._is_packed_list(self):
            self = stack(g, self, g.op("Constant", value_t=torch.tensor([0])))
        return min(g, self)
    return min(g, self, other)


@_onnx_symbolic("prim::data")
@_beartype.beartype
def prim_data(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("prim::layout")
def prim_layout(g: jit_utils.GraphContext, self):
    # Always return 'torch.strided'. Other layout types are not supported by JIT 'TensorType'.
    # Layout class defined in 'c10/core/Layout.h'.
    return g.op("Constant", value_t=torch.tensor(0))


@_onnx_symbolic("prim::ListConstruct")
@_beartype.beartype
def prim_list_construct(g: jit_utils.GraphContext, *inputs, **kwargs):
    return None


@_onnx_symbolic("prim::ListUnpack")
@_beartype.beartype
def prim_list_unpack(
    g: jit_utils.GraphContext, *inputs, **kwargs
) -> Optional[List[_C.Value]]:
    if len(inputs) == 1 and inputs[0].node().kind() == "prim::ListConstruct":
        # Cancel the previous node if it is ListConstruct by returning its inputs
        # TODO(justinchuby): Use a public method in the helper module
        return symbolic_helper._unpack_list(inputs[0])

    return None


@_onnx_symbolic("prim::TupleConstruct")
@_beartype.beartype
def prim_tuple_construct(g: jit_utils.GraphContext, *inputs, **kwargs):
    return None


@_onnx_symbolic("prim::Uninitialized")
@_beartype.beartype
def prim_uninitialized(g: jit_utils.GraphContext, *inputs, **kwargs):
    return None


# exists to refine the type of the Value
# if x is an optional Tensor, unchecked_cast will cast
# x to Tensor, so the rest of the graph knows that x is a Tensor
# this doesn't do anything in runtime and is a noop in ONNX
@_onnx_symbolic("prim::unchecked_cast")
@_beartype.beartype
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("prim::dtype")
@_beartype.beartype
def prim_dtype(g: jit_utils.GraphContext, self):
    scalar_type = symbolic_helper._try_get_scalar_type(self)
    if scalar_type is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    # This node records a torch dtype as int
    return g.op("Constant", value_t=torch.tensor(scalar_type))


@_onnx_symbolic("prim::tolist")
@_beartype.beartype
def prim_tolist(g: jit_utils.GraphContext, input, dim_val, elem_ty_val):
    """tolist is currently supported only for 1D input tensors.

    dim_val and elem_ty_val represent dimension and type annotations
    that need to match dimension and type of the input tensor.
    """
    dim = symbolic_helper._maybe_get_const(dim_val, "i")
    if dim > 1:
        return symbolic_helper._unimplemented("prim::tolist", "dim_val > 1", input)
    return input


# -----------------------------------------------------------------------------
# Symbolic functions that need extra context
# -----------------------------------------------------------------------------
@_onnx_symbolic("prim::device")
@_beartype.beartype
def prim_device(g: jit_utils.GraphContext, *inputs, **kwargs) -> None:
    output_type = g.original_node.output().type()
    if isinstance(output_type, _C.DeviceObjType):
        return None

    return symbolic_helper._unimplemented(
        "prim::device",
        f"output type should be 'DeviceObjType', not '{output_type.kind()}'",
        g.original_node.output(),
    )


@_onnx_symbolic("prim::Loop")
@_beartype.beartype
def prim_loop(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    node = g.original_node
    env = g.env
    values_in_env = g.values_in_env
    params_dict = g.params_dict

    operator_export_type = GLOBALS.operator_export_type
    opset_version = GLOBALS.export_onnx_opset_version

    old_blocks = tuple(node.blocks())
    new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(
        g, "Loop", *inputs, outputs=node.outputsSize(), n_blocks=len(old_blocks)
    )

    for old_block, new_block_context in zip(old_blocks, new_block_contexts):
        # Copy input metadata to subblock
        #
        #   prim::Loop(iter, cond, input_1, ..., input_n)
        #     block0(iter, input_1, ..., input_n)
        #
        # For `Loop` node, copy metadata for `iter`, `input_1`, ..., `input_n`.
        for i, b_in in enumerate(old_block.inputs()):
            if i == 0 and i < len(inputs):
                b_in.setType(inputs[i].type())
            # For optional block inputs, they may switch between None not-None inside
            # the loop body, so if the loop input is not optional, the block input may
            # still need to be optional.
            if (
                i > 0
                and (i + 1) < len(inputs)
                and not isinstance(b_in.type(), _C.OptionalType)
            ):
                b_in.setType(inputs[i + 1].type())
        torch._C._jit_pass_onnx_block(
            old_block,
            new_block_context.block,
            operator_export_type,
            env,
            values_in_env,
            False,
        )
    fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
        new_node, opset_version
    )
    # Run shape type inference for Loop after subblock is converted.
    if GLOBALS.onnx_shape_inference:
        torch._C._jit_pass_onnx_node_shape_type_inference(
            new_node, params_dict, opset_version
        )
    return fixed_outputs


@_onnx_symbolic("prim::If")
@_beartype.beartype
def prim_if(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    n = g.original_node
    block = g.block
    env = g.env
    values_in_env = g.values_in_env
    params_dict = g.params_dict

    operator_export_type = GLOBALS.operator_export_type
    opset_version = GLOBALS.export_onnx_opset_version

    static_if = inputs[0].node().kind() == "onnx::Constant"
    if static_if:
        # Fold static if
        #
        # The torch IR
        # graph(%embedding_matrix.1 : Float(10, 15, strides=[15, 1], requires_grad=0, device=cpu),
        #    %input.1 : Long(6, strides=[1], requires_grad=0, device=cpu), ...
        # %65 : Bool(requires_grad=0, device=cpu) = prim::Constant[value={0}]()
        # %21 : Long(device=cpu) = aten::eq(%20, %64)
        # %22 : Long(device=cpu) = prim::If(%21)
        #     block0():
        #     %23 : Long(device=cpu) = aten::is_floating_point(%input.1)
        #     -> (%23)
        #     block1():
        #     -> (%65)
        # %input.53 : Tensor, %weight : Tensor = prim::If(%22)
        #     block0():
        #     -> (%embedding_matrix.1, %input.1)
        #     block1():
        #     -> (%input.1, %embedding_matrix.1)
        # %26 : int[] = aten::size(%input.53)
        #
        # The converted ONNX graph
        # %10 : Bool(device=cpu) = onnx::Constant[value={0}]()
        # %14 : Bool(device=cpu) = onnx::Equal(%13, %8)
        # %15 : Bool(requires_grad=0, device=cpu) = onnx::Constant[value={0}]()
        # %16 : Long(1, strides=[1], device=cpu) = onnx::Shape(%input.1)
        input_flag = symbolic_helper._node_get(inputs[0].node(), "value").tolist()
        const_value = (
            all(input_flag) if isinstance(input_flag, list) else bool(input_flag)
        )
        block_idx = 0 if const_value else 1
        current_b = list(n.blocks())[block_idx]
        env = torch._C._jit_pass_onnx_block(
            current_b,
            block,
            operator_export_type,
            env,
            values_in_env,
            True,
        )
        if_output_list = list(n.outputs())
        current_b_list = list(current_b.outputs())

        final_b_list = []
        for idx in range(len(if_output_list)):
            if current_b_list[idx] not in env:
                raise errors.SymbolicValueError(
                    f"The sub block ATen output {current_b_list[idx]} is not in env.",
                    current_b_list[idx],
                )  # type:ignore[operator]
            onnx_b = env[current_b_list[idx]]
            final_b_list.append(onnx_b)
        return final_b_list
    else:
        old_blocks = tuple(n.blocks())
        new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(
            g, "If", *inputs, outputs=n.outputsSize(), n_blocks=len(old_blocks)
        )

        for old_block, new_block_context in zip(old_blocks, new_block_contexts):
            torch._C._jit_pass_onnx_block(
                old_block,
                new_block_context.block,
                operator_export_type,
                env,
                values_in_env,
                False,
            )
        fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
            new_node, opset_version
        )
        # Run shape type inference for If after subblock is converted.
        if GLOBALS.onnx_shape_inference:
            torch._C._jit_pass_onnx_node_shape_type_inference(
                new_node, params_dict, opset_version
            )
        return fixed_outputs


@_onnx_symbolic("prim::Constant")
@_beartype.beartype
def prim_constant(g: jit_utils.GraphContext, *inputs, **attrs):
    node = g.original_node

    if node.mustBeNone():
        return None
    # This must go before checking for string values, because some device constants
    # have string values, but we want to keep them as unconverted Device types so
    # that eq() can work on them.
    if isinstance(node.output().type(), _C.DeviceObjType):
        return None
    if node.kindOf("value") == "t":
        return g.op("Constant", value_t=symbolic_helper._node_get(node, "value"))
    if node.kindOf("value") == "s":
        return g.op("Constant", value_s=symbolic_helper._node_get(node, "value"))
    if node.output().type().isSubtypeOf(
        _C.ListType.ofInts()
    ) or node.output().type().isSubtypeOf(_C.ListType.ofFloats()):
        return g.op(
            "Constant", value_t=torch.tensor(symbolic_helper._node_get(node, "value"))
        )
    if node.output().type().isSubtypeOf(_C.ListType.ofStrings()):
        str_constants = [
            g.op("Constant", value_s=s)
            for s in symbolic_helper._node_get(node, "value")
        ]
        return g.op("prim::ListConstruct", *str_constants)

    raise errors.SymbolicValueError(
        f"Unsupported prim::Constant kind: '{node.kindOf('value')}'. "
        f"Please send a bug report at {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
        node.output(),
    )


@_onnx_symbolic("prim::type")
@_beartype.beartype
def prim_type(g: jit_utils.GraphContext, device_value: _C.Value, *args, **kwargs):
    if device_value.node().kind() == "prim::device":
        device = jit_utils.get_device_from_value(device_value.node().input())
        if device is not None:
            return g.op("Constant", value_s=str(device))

    return symbolic_helper._unimplemented(
        "prim::type",
        "Device type cannot be statically determined.",
        device_value,
    )


@_onnx_symbolic("onnx::Placeholder")
@_beartype.beartype
def onnx_placeholder(g: jit_utils.GraphContext, *inputs, **attrs):
    node = g.original_node
    block = g.block
    env = g.env
    values_in_env = g.values_in_env

    return torch._C._jit_onnx_convert_pattern_from_subblock(
        block, node, env, values_in_env
    )


@_onnx_symbolic("aten::resolve_conj")
@_onnx_symbolic("aten::resolve_neg")
@_beartype.beartype
def noop_complex_operators(g: jit_utils.GraphContext, input: _C.Value):
    # ONNX does not have operators to *directly* manipulate real/imaginary components
    # However, a few torch APIs (e.g. .tolist()) use complex operations when input is real,
    # which results in failures due to missing operators for complex numbers

    # `aten::resolve_conj` and `aten::resolve_neg` can safely be implemented as no-op
    return input


@_onnx_symbolic("aten::_conj")
@_onnx_symbolic("aten::conj_physical")
@_beartype.beartype
def unsupported_complex_operators(g: jit_utils.GraphContext, input: _C.Value):
    # ONNX does not have operators to *directly* manipulate real/imaginary components
    # However, a few torch APIs (e.g. .tolist()) use complex operations when input is real,
    # which results in failures due to missing operators for complex numbers

    # While `aten::_conj` and `aten::conj_physical` raise exception when input is complex
    if symbolic_helper.is_complex_value(input):
        # FIXME(justinchuby): report correct name for symbolic being executed
        return symbolic_helper._onnx_unsupported(
            "aten::_conj, aten::conj_physical",
            input,
        )

    # they can safely be implemented as no-op for real numbers only
    return noop_complex_operators(g, input)


@_onnx_symbolic("aten::logit")
@_beartype.beartype
def logit(g: jit_utils.GraphContext, self: torch._C.Value, eps: torch._C.Value):
    one = g.op("Constant", value_t=torch.tensor(1.0))

    if not symbolic_helper._is_none(eps):
        eps = g.op(
            "Cast", eps, to_i=_type_utils.JitScalarType.from_value(self).onnx_type()
        )
        one_sub_eps = g.op("Sub", one, eps)
        self_less_equal_one_sub_eps = g.op("Greater", one_sub_eps, self)
        temporary_self = g.op("Where", self_less_equal_one_sub_eps, self, one_sub_eps)

        temporary_self_less_eps = g.op("Less", temporary_self, eps)
        z = g.op("Where", temporary_self_less_eps, eps, temporary_self)
    else:
        z = self

    sub = g.op("Sub", one, z)
    div = g.op("Div", z, sub)
    return g.op("Log", div)
