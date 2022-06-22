"""This file exports ONNX ops for opset 9.

Opset 9 is supported by ONNX release 1.4.1
release on 01/23/19
"""

import functools
import math
import sys
import warnings
from typing import List, Optional

import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx import _patch_torch  # noqa: F401
from torch.onnx import symbolic_helper
from torch.onnx._exporter_states import (
    SymbolicContext,  # Special case class import for readability
)
from torch.onnx._globals import GLOBALS

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# Note [Pointwise by scalar]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# What happens if you add a tensor with a constant (e.g., x + 2)?  There are
# some moving parts to implementing the ONNX translation in this case:
#
#   - By the time we get the scalar in a symbolic function here, it is no longer
#     a Python long/float, but a PyTorch tensor with numel == 1 (eventually, we
#     want it to be a zero dim tensor but this change has not happened yet.)
#     However, the type of this scalar is *exactly* what the user wrote in
#     Python, which may not match the tensor it is being added to.  PyTorch
#     will do implicit conversions on scalars; however, ONNX will not, so
#     we must do the conversion ourselves.  This is what _if_scalar_type_as
#     does.
#
#   - Dispatch to these functions takes advantage an outrageous coincidence
#     between the tensor and scalar name.  When we add two tensors together,
#     you get the dispatch:
#
#       add(*[self, other], **{"alpha": alpha})
#
#     When you add a tensor and a scalar, you get the dispatch:
#
#       add(*[self], **{"other": other, "alpha": alpha})
#
#     By having the argument name line up with the name of the scalar attribute
#     if it exists, we can write a single function for both overloads.
#

__all__ = [
    "unused",
    "reshape",
    "reshape_as",
    "add",
    "sub",
    "rsub",
    "mul",
    "div",
    "addcmul",
    "floor_divide",
    "floordiv",
    "true_divide",
    "reciprocal",
    "cat",
    "stack",
    "mm",
    "bmm",
    "matmul",
    "addmm",
    "neg",
    "sqrt",
    "rsqrt",
    "tanh",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sigmoid",
    "sign",
    "overload_by_arg_count",
    "sum",
    "mean",
    "prod",
    "cumsum",
    "t",
    "expand",
    "expand_as",
    "embedding",
    "embedding_bag",
    "size",
    "transpose",
    "permute",
    "view",
    "view_as",
    "unsafe_chunk",
    "split",
    "unsafe_split",
    "split_with_sizes",
    "unsafe_split_with_sizes",
    "unbind",
    "select",
    "square",
    "squeeze",
    "prelu",
    "silu",
    "mish",
    "op_with_optional_float_cast",
    "relu",
    "relu6",
    "ceil",
    "floor",
    "threshold",
    "leaky_relu",
    "glu",
    "softmax",
    "softplus",
    "get_pool_ceil_padding",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "constant_pad_nd",
    "reflection_pad",
    "replication_pad",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "pad",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_linear1d",
    "upsample_bilinear2d",
    "upsample_trilinear3d",
    "bitwise_not",
    "wrap_logical_op_with_cast_to",
    "wrap_logical_op_with_cast_to_and_from",
    "wrap_logical_op_with_negation",
    "eq",
    "ne",
    "gt",
    "gt_impl",
    "lt",
    "lt_impl",
    "ge",
    "le",
    "logical_and",
    "logical_or",
    "logical_xor",
    "where",
    "log_softmax",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "batch_norm",
    "layer_norm",
    "instance_norm",
    "unfold",
    "elu",
    "selu",
    "index_select",
    "index_put",
    "index_fill",
    "index_copy",
    "bucketize",
    "type_as",
    "cosine_similarity",
    "pairwise_distance",
    "clone",
    "abs",
    "log",
    "log1p",
    "log10",
    "pow",
    "clamp",
    "clamp_min",
    "clamp_max",
    "max",
    "maximum",
    "min",
    "minimum",
    "amax",
    "amin",
    "aminmax",
    "exp",
    "dropout",
    "feature_dropout",
    "alpha_dropout",
    "feature_alpha_dropout",
    "dropout_",
    "feature_dropout_",
    "alpha_dropout_",
    "feature_alpha_dropout_",
    "norm",
    "conv_tbc",
    "empty",
    "empty_like",
    "new_empty",
    "scalar_tensor",
    "tensor",
    "as_tensor",
    "zeros",
    "zeros_like",
    "new_zeros",
    "ones",
    "ones_like",
    "new_ones",
    "full",
    "full_like",
    "new_full",
    "eye",
    "slice",
    "hardtanh",
    "hardswish",
    "hardsigmoid",
    "tanhshrink",
    "hardshrink",
    "softshrink",
    "alias",
    "unsqueeze",
    "sort",
    "numel",
    "topk",
    "to",
    "repeat",
    "repeat_interleave",
    "pixel_shuffle",
    "pixel_unshuffle",
    "lstm",
    "lstm_cell",
    "gru",
    "rnn_tanh",
    "rnn_relu",
    "detach",
    "contiguous",
    "randn",
    "rand",
    "randn_like",
    "rand_like",
    "rrelu",
    "bernoulli",
    "log_sigmoid",
    "erf",
    "flatten",
    "nonzero",
    "nonzero_numpy",
    "isnan",
    "narrow",
    "argmax",
    "argmin",
    "scatter",
    "scatter_add",
    "log2",
    "is_floating_point",
    "one_hot",
    "gather",
    "std",
    "var",
    "var_mean",
    "std_mean",
    "logsumexp",
    "arange",
    "linspace",
    "lift",
    "masked_fill",
    "index",
    "linalg_norm",
    "linalg_vector_norm",
    "linalg_matrix_norm",
    "linalg_cross",
    "frobenius_norm",
    "multinomial",
    "baddbmm",
    "meshgrid",
    "remainder",
    "gelu",
    "group_norm",
    "dim",
    "item",
    "take",
    "kl_div",
    "as_strided",
    "linear",
    "hann_window",
    "mv",
    "dot",
    "movedim",
    "fill",
    "index_add",
    "roll",
    "cross",
    "cdist",
    "lerp",
    "broadcast_tensors",
    "Prim",
    "Onnx",
]

# used to represent "missing" optional inputs
def unused(g):
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


def _shape_as_tensor(g, input):
    return g.op("Shape", input)


def _reshape_from_tensor(g, input, shape):
    if isinstance(shape, list):
        shape = g.op("Concat", *shape, axis_i=0)
    return reshape(g, input, shape)


def reshape(g, self, shape):
    return symbolic_helper._reshape_helper(g, self, shape)


def reshape_as(g, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


def add(g, self, other, alpha=None):
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Add", 9, 11, "Add between list of tensors not supported"
        )

    # default alpha arg is to allow no-alpha add (aten add st overload no alpha)
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        return symbolic_helper._unimplemented("add", "alpha != 1")
    return g.op("Add", self, other)


def sub(g, self, other, alpha=None):
    # default alpha arg is to allow no-alpha sub (aten sub st overload no alpha)
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        return symbolic_helper._unimplemented("sub", "alpha != 1")
    return g.op("Sub", self, other)


def rsub(g, self, other, alpha=None):
    return sub(g, other, self, alpha=alpha)


def mul(g, self, other):
    return g.op("Mul", self, other)


def div(g, self, other, *args):
    if len(args) == 0:
        return true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@symbolic_helper.parse_args("v", "v", "v", "f")
def addcmul(g, self, tensor1, tensor2, value=1.0):
    value_tens = g.op("Constant", value_t=torch.tensor([value]))
    return add(g, self, mul(g, mul(g, tensor1, tensor2), value_tens))


@symbolic_helper.parse_args("v", "v", "s")
def _div_rounding_mode(g, self, other, rounding_mode):
    if rounding_mode is None:
        return true_divide(g, self, other)
    elif rounding_mode == "floor":
        return _floor_divide(g, self, other)
    elif rounding_mode == "trunc":
        return _trunc_divide(g, self, other)
    else:
        raise RuntimeError(
            f'Unsupported rounding mode: "{rounding_mode}". Expected None, "floor" or "trunc"'
        )


def _trunc_divide(g, self, other):
    out = g.op("Div", self, other)
    # the correct operation is truncate, which is not supported in ONNX,
    # we cannot call floor since it will behave differently for negative numbers
    # (eg. -0.1 should become -0 )
    # - if scalar_type information are not available, assume that
    # we need to call floor (treat as float)
    out = g.op("Cast", out, to_i=symbolic_helper.cast_pytorch_to_onnx["Long"])

    # Matching PyTorch's behavior:
    # - if self is fp the output's type is self's type
    # - if self is not fp and other is fp, the output is of type "Float"
    # - self is not fp and other is not fp, the output's type is self's output type
    # - the output type defaults to Float
    scalar_type = self.type().scalarType()

    if scalar_type is not None:
        if (
            not symbolic_helper._is_fp(self)
            and other.type().scalarType() is not None
            and symbolic_helper._is_fp(other)
        ):
            out = g.op("Cast", out, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
        else:
            out = g.op(
                "Cast", out, to_i=symbolic_helper.cast_pytorch_to_onnx[scalar_type]
            )
    else:
        out = g.op("Cast", out, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
    return out


def _floor_divide(g, self, other):
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


def floor_divide(g, self, other):
    # Deprecated behavior, floor_divide actually truncates
    return _trunc_divide(g, self, other)


def floordiv(g, self, other):
    return floor_divide(g, self, other)


def true_divide(g, self, other):
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
    onnx_scalar_type = symbolic_helper.cast_pytorch_to_onnx["Float"]
    assert scalar_type is torch.float or scalar_type is torch.double
    if torch.get_default_dtype() is torch.double:
        onnx_scalar_type = symbolic_helper.cast_pytorch_to_onnx["Double"]

    self = g.op("Cast", self, to_i=onnx_scalar_type)
    other = g.op("Cast", other, to_i=onnx_scalar_type)
    return g.op("Div", self, other)


def reciprocal(g, self):
    # torch.reciprocal implicitly casts to float, so we do the same.
    if not symbolic_helper._is_fp(self):
        self = g.op("Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
    return g.op("Reciprocal", self)


@symbolic_helper.parse_args("v", "i")
def cat(g, tensor_list, dim):
    tensors = symbolic_helper._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@symbolic_helper.parse_args("v", "i")
def stack(g, tensor_list, dim):
    unsqueezed = [
        symbolic_helper._unsqueeze_helper(g, t, [dim])
        for t in symbolic_helper._unpack_list(tensor_list)
    ]
    return g.op("Concat", *unsqueezed, axis_i=dim)


def _list(g, self):
    return self


def mm(g, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    C = g.op("Constant", value_t=torch.tensor([1]))
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


def bmm(g, self, other):
    return g.op("MatMul", self, other)


def matmul(g, self, other):
    return g.op("MatMul", self, other)


@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g, self, mat1, mat2, beta, alpha):
    dtype = None
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    mat1_dtype = symbolic_helper._try_get_scalar_type(mat1)
    mat2_dtype = symbolic_helper._try_get_scalar_type(mat2)
    if self_dtype is not None:
        dtype = self_dtype
    elif mat1_dtype is not None:
        dtype = mat1_dtype
    elif mat2_dtype is not None:
        dtype = mat2_dtype

    mat1_rank = symbolic_helper._get_tensor_rank(mat1)
    mat2_rank = symbolic_helper._get_tensor_rank(mat2)

    def isNotNoneAnd(v, u):
        return v is not None and v != u

    if dtype is not None and (isNotNoneAnd(mat1_rank, 2) or isNotNoneAnd(mat2_rank, 2)):
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
        dtype = symbolic_helper.scalar_type_to_pytorch_type[dtype]

        res1 = g.op("MatMul", mat1, mat2)
        res2 = self

        alpha = symbolic_helper._scalar(alpha)
        beta = symbolic_helper._scalar(beta)

        if alpha != 1:
            alpha = g.op("Constant", value_t=torch.tensor(alpha, dtype=dtype))
            res1 = g.op("Mul", res1, alpha)
        if beta != 1:
            beta = g.op(
                "Constant",
                value_t=torch.tensor(symbolic_helper._scalar(beta), dtype=dtype),
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


def neg(g, self):
    return g.op("Neg", self)


def sqrt(g, self):
    return g.op("Sqrt", self)


def rsqrt(g, self):
    return g.op(
        "Div", symbolic_helper._if_scalar_type_as(g, torch.ones(1), self), sqrt(g, self)
    )


def tanh(g, self):
    return g.op("Tanh", self)


def sin(g, self):
    return g.op("Sin", self)


def cos(g, self):
    return g.op("Cos", self)


def tan(g, self):
    return g.op("Tan", self)


def asin(g, self):
    return g.op("Asin", self)


def acos(g, self):
    return g.op("Acos", self)


def atan(g, self):
    return g.op("Atan", self)


# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qsigmoid.cpp
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
def sigmoid(g, self):
    return g.op("Sigmoid", self)


def sign(g, self):
    return g.op("Sign", self)


def _slice(g, input, axes, starts, ends):
    assert len(starts) == len(ends)
    if len(starts) == 1 and starts[0] == 0 and ends[0] == 9223372036854775807:
        return input
    return g.op("Slice", input, axes_i=axes, starts_i=starts, ends_i=ends)


def _maybe_cast_reduce_op_input(g, self):
    dtype = self.type().scalarType()
    # This check only covers traced modules where dtype is present
    if dtype is not None:
        # pytorch reduce-ops cast all other integral types to int64
        if not symbolic_helper._is_fp(self) and not (dtype == "Long"):
            self = _cast_Long(g, self, False)  # type: ignore[name-defined]
    return self


def _reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return symbolic_helper._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            # dim-reduce path
            desc = "is" if allow_multi_dim_support else "i"
            dim, keepdim = symbolic_helper._get_const(
                dim, desc, "dim"
            ), symbolic_helper._get_const(keepdim, "i", "keepdim")
            dim_list = dim if allow_multi_dim_support else [dim]
            return g.op(onnx_op_name, self, axes_i=dim_list, keepdims_i=keepdim)

    return symbolic


def overload_by_arg_count(fn):
    @functools.wraps(fn)
    def wrapper(g, *args):
        overloads = fn(g, *args)
        last_exception = None
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        raise NotImplementedError(f"Unknown aten::{fn.__name__} signature")

    return wrapper


def _reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True):
    symbolic = _reduce_op_symbolic(
        onnx_op, allow_multi_dim_support=allow_multi_dim_support
    )

    @overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @symbolic_helper.parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                self = g.op(
                    "Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
                )
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype")
            return symbolic(g, self)

        dim_desc = "is" if allow_multi_dim_support else "i"

        @symbolic_helper.parse_args("v", dim_desc, "i", "none")
        def reduce_dim(g, self, dim, keepdim, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                self = g.op(
                    "Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
                )
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype")
            return symbolic(g, self, dim, keepdim)

        return reduce_nodim, reduce_dim

    return reduce


sum = _reduce_with_dtype("ReduceSum", "sum")
mean = _reduce_with_dtype("ReduceMean", "mean")
# torch.prod does not support multidimensional "dim"
prod = _reduce_with_dtype("ReduceProd", "prod", allow_multi_dim_support=False)


@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g, input, dim, dtype):
    if symbolic_helper.is_caffe2_aten_fallback():
        if dtype.node().kind() != "prim::Constant":
            return symbolic_helper._unimplemented(name, "dtype")
        return g.at("cumsum", input, dim_i=dim)
    else:
        symbolic_helper._onnx_opset_unsupported("cumsum", 9, 11)


def _sample_dirichlet(g, self, generator):
    if symbolic_helper.is_caffe2_aten_fallback():
        if not symbolic_helper._is_none(generator):
            return symbolic_helper._unimplemented(
                "_sample_dirichlet", "We are not able to export generator"
            )
        return g.at("_sample_dirichlet", self)
    else:
        return symbolic_helper._onnx_unsupported("_sample_dirichlet")


def _standard_gamma(g, self, generator):
    if symbolic_helper.is_caffe2_aten_fallback():
        if not symbolic_helper._is_none(generator):
            return symbolic_helper._unimplemented(
                "_standard_gamma", "We are not able to export generator"
            )
        return g.at("_standard_gamma", self)
    else:
        return symbolic_helper._onnx_unsupported("_standard_gamma")


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def expand(g, self, size, implicit):
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
    dtype = symbolic_helper.ScalarType.INT64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


def expand_as(g, self, other):
    self_t = symbolic_helper._maybe_get_const(self, "t")
    if isinstance(self_t, torch.Tensor):
        orig_type = self_t.dtype
        self_t = self_t.to(torch.double)
        dims = []
        for d in range(self_t.dim()):
            if torch.equal(self_t.mean(d).unsqueeze(d).expand_as(self_t), self_t):
                dims.append(d)
                self = g.op("Constant", value_t=self_t.mean(dims).to(orig_type))

    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


@symbolic_helper.parse_args("v", "v", "i", "b", "v")
def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    if scale_grad_by_freq and GLOBALS.export_training:
        raise RuntimeError(
            "Unsupported: ONNX export of embedding with scale_grad_by_freq=True "
            "for training mode. ONNX does not support scaling the gradients."
        )
    if padding_idx >= 0 and GLOBALS.export_training:
        warnings.warn(
            "Warning: ONNX export of embedding with padding_idx >= 0 "
            "for training mode. "
            "ONNX does not support not updating the embedding vector at padding_idx during training."
        )

    return g.op("Gather", weight, indices)


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
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
):
    if not symbolic_helper._is_none(per_sample_weights):
        return symbolic_helper._onnx_unsupported(
            "embedding_bag  with per_sample_weights"
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
    else:
        return symbolic_helper._onnx_unsupported("embedding_bag")


def size(g, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    if symbolic_helper._maybe_get_const(dim, "i") < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            dim = symbolic_helper._maybe_get_const(dim, "i") + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    return symbolic_helper._size_helper(g, self, dim)


@symbolic_helper.parse_args("v", "i", "i")
def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return g.op("Transpose", self, perm_i=axes)
    else:
        # if we don't have dim information we cannot
        # output a permute so use ATen instead
        if symbolic_helper.is_caffe2_aten_fallback():
            return g.at(
                "transpose", self, overload_name="int", dim0_i=dim0, dim1_i=dim1
            )
        else:
            raise RuntimeError(
                "Unsupported: ONNX export of transpose for tensor " "of unknown rank."
            )


@symbolic_helper.parse_args("v", "is")
def permute(g, self, dims):
    if dims == list(range(0, len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


def view(g, self, size):
    return reshape(g, self, size)


def view_as(g, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@symbolic_helper.parse_args("v", "i", "i", "i")
def unsafe_chunk(g, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unsafe_chunk", 9, 11, "Dynamic number of outputs not supported"
        )
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented("unsafe_chunk", "unknown dimension size")
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@symbolic_helper.parse_args("v", "v", "v", "i")
def split(g, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split", 9, 11, "Dynamic number of outputs not supported"
        )
    split_val = split_size_or_sizes.node()["value"]
    if split_val.dim() > 0:
        return split_with_sizes(g, self, split_size_or_sizes, dim, _outputs)
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")
    dim = symbolic_helper._get_const(dim, "i", "dim")

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "split", 9, 11, "Unknown dimension size not supported"
            )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


def unsafe_split(g, self, split_size_or_sizes, dim, _outputs=None):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@symbolic_helper.parse_args("v", "is", "i", "i")
def split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split_with_sizes", 9, 11, "Dynamic number of outputs not supported"
        )
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=_outputs)


def unsafe_split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@symbolic_helper.parse_args("v", "i", "i")
def unbind(g, self, dim=0, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unbind", 9, 11, "Dynamic number of outputs not supported"
        )

    outputs = g.op("Split", self, split_i=[1] * _outputs, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [
        symbolic_helper._squeeze_helper(g, out, [dim]) for out in outputs
    ]
    return squeezed_outputs


@symbolic_helper.parse_args("v", "i", "v")
def select(g, self, dim, index):
    index = symbolic_helper._maybe_get_scalar(index)
    if (not symbolic_helper._is_value(index)) and (index < 0):
        if index == -1:
            end_index = 9223372036854775807
        else:
            end_index = index + 1
        slice_node = symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[index], ends=[end_index]
        )
        return symbolic_helper._squeeze_helper(g, slice_node, [dim])
    else:
        return g.op("Gather", self, index, axis_i=dim)


def square(g, self):
    return g.op("Mul", self, self)


def squeeze(g, self, dim=None):
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
                "squeeze", "negative axis with unknown input rank"
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


def prelu(g, self, weight):
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


def silu(g, input):
    return g.op("Mul", input, g.op("Sigmoid", input))


def mish(g, input):
    return g.op("Mul", input, g.op("Tanh", g.op("Softplus", input)))


def op_with_optional_float_cast(g, op_name, *args, **kwargs):
    """Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, "Float" by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    """
    opset_before = kwargs.pop("opset_before", None)
    target_float_t = kwargs.pop("target_float_t", "Float")

    inputs = list(args)
    dtype_0 = inputs[0].type().scalarType()

    require_cast = not symbolic_helper._is_fp(inputs[0]) and (
        opset_before is None or GLOBALS.export_onnx_opset_version < opset_before
    )

    if require_cast:
        for input in inputs:
            if input.isCompleteTensor() and input.type().scalarType() != dtype_0:
                raise RuntimeError(
                    f"Inputs of {op_name} must have same dtype. Got {dtype_0} and {input.type().scalarType()}"
                )
        for i, input in enumerate(inputs):
            if input.isCompleteTensor() and not symbolic_helper._is_fp(input):
                inputs[i] = g.op(
                    "Cast",
                    input,
                    to_i=symbolic_helper.cast_pytorch_to_onnx[target_float_t],
                )

    self = g.op(op_name, *inputs, **kwargs)

    if require_cast:
        self = g.op("Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype_0])

    return self


@symbolic_helper.quantized_args(True)
def relu(g, input):
    return op_with_optional_float_cast(g, "Relu", input, opset_before=14)


@symbolic_helper.quantized_args(True)
def relu6(g, input):
    relu = op_with_optional_float_cast(g, "Relu", input, opset_before=14)
    return clamp_max(g, relu, 6)


def ceil(g, input):
    return g.op("Ceil", input)


def floor(g, input):
    return g.op("Floor", input)


def _len(g, self):
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@symbolic_helper.parse_args("v", "t", "t")
def threshold(g, self, threshold, value):
    # See Note [Export inplace]
    if symbolic_helper._scalar(threshold) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero threshold")
    if symbolic_helper._scalar(value) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero value")
    return g.op("Relu", self)


def leaky_relu(g, input, negative_slope, inplace=False):
    negative_slope = symbolic_helper._get_const(negative_slope, "t", "negative_slope")
    # See Note [Export inplace]
    # TODO: Talk to ONNX about unconditional cast of scalar to float
    return g.op("LeakyRelu", input, alpha_f=symbolic_helper._scalar(negative_slope))


@symbolic_helper.parse_args("v", "i")
def glu(g, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", input, axis_i=dim, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))


@symbolic_helper.parse_args("v", "i", "none")
def softmax(g, input, dim, dtype=None):
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
                "Cast", softmax, to_i=symbolic_helper.scalar_type_to_onnx[parsed_dtype]
            )

        if is_transpose_required:
            softmax = g.op("Transpose", softmax, perm_i=axes)
        return softmax

    # Apply max normalization.
    input = g.op("Sub", input, g.op("ReduceMax", input, axes_i=[dim], keepdims_i=1))

    exp = g.op("Exp", input)
    sum = symbolic_helper._reducesum_helper(g, exp, axes_i=[dim])
    softmax = g.op("Div", exp, sum)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=symbolic_helper.scalar_type_to_onnx[parsed_dtype]
        )
    return softmax


def softplus(g, self, beta, threshold):
    beta_const = symbolic_helper._maybe_get_const(beta, "f")
    if beta_const != 1:
        return g.op("Div", g.op("Softplus", g.op("Mul", self, beta)), beta)
    return g.op("Softplus", self)


def get_pool_ceil_padding(input, kernel_size, stride, padding):
    sizes = symbolic_helper._get_tensor_sizes(input)
    dim = sizes[-len(padding) :] if sizes is not None else None
    if dim is None or any([i is None for i in dim]):
        return symbolic_helper._unimplemented(name, "input size not accessible")
    ceiled_output_dim = [
        int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i])))
        + 1
        for i in range(0, len(padding))
    ]
    # ensure last pooling starts inside
    ceiled_output_dim = [
        ceiled_output_dim[i] - 1
        if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
        else ceiled_output_dim[i]
        for i in range(0, len(ceiled_output_dim))
    ]
    padding_ceil = [
        0
        if (stride[i] == 1)
        else (
            kernel_size[i]
            - (dim[i] + 2 * padding[i] - ((ceiled_output_dim[i] - 1) * stride[i] + 1))
        )
        for i in range(0, len(padding))
    ]
    # ensure padding is not > kernel_size
    padding_ceil = [
        (
            int(padding_ceil[i])
            if padding_ceil[i] < kernel_size[i] - 1
            else int(kernel_size[i] - 1)
        )
        if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
        else int(padding_ceil[i])
        for i in range(0, len(padding_ceil))
    ]
    return padding_ceil


def _max_pool(name, tuple_fn, ndims, return_indices):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if set(tuple_fn(dilation)) != {1}:
            return symbolic_helper._unimplemented(name, "dilation")
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
        # Using this tensor as a reference, we extract the first index of each axis and substract
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
                starts=tuple_fn(0),
                ends=tuple_fn(1),
            )
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d = _max_pool(
    "max_pool1d", torch.nn.modules.utils._single, 1, return_indices=False
)
max_pool2d = _max_pool(
    "max_pool2d", torch.nn.modules.utils._pair, 2, return_indices=False
)
max_pool3d = _max_pool(
    "max_pool3d", torch.nn.modules.utils._triple, 3, return_indices=False
)
max_pool1d_with_indices = _max_pool(
    "max_pool1d_with_indices",
    torch.nn.modules.utils._single,
    1,
    return_indices=True,
)
max_pool2d_with_indices = _max_pool(
    "max_pool2d_with_indices",
    torch.nn.modules.utils._pair,
    2,
    return_indices=True,
)
max_pool3d_with_indices = _max_pool(
    "max_pool3d_with_indices",
    torch.nn.modules.utils._triple,
    3,
    return_indices=True,
)


def _avg_pool(name, tuple_fn):
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=None,
    ):
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
        if count_include_pad:
            input = g.op(
                "Pad",
                input,
                pads_i=((0,) * 2 + padding) * 2,
                mode_s="constant",
                value_f=0.0,
            )
            padding = (0,) * len(padding)
        if ceil_mode:
            padding = padding + tuple(a + b for (a, b) in zip(padding_ceil, padding))
        else:
            padding = padding * 2
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=padding,
        )
        return output

    return symbolic_fn


avg_pool1d = _avg_pool("avg_pool1d", torch.nn.modules.utils._single)
avg_pool2d = _avg_pool("avg_pool2d", torch.nn.modules.utils._pair)
avg_pool3d = _avg_pool("avg_pool3d", torch.nn.modules.utils._triple)


def _adaptive_pool(name, type, tuple_fn, fn=None):
    @symbolic_helper.quantized_args(True, False)
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
        try:
            output_size = symbolic_helper._parse_arg(output_size, "is")
        except Exception:
            return symbolic_helper._onnx_unsupported(
                "adaptive pooling, since output_size is not constant."
            )
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        sizes = symbolic_helper._get_tensor_sizes(input)
        try:
            dim = sizes[2:]
        except Exception:
            dim = None
        if dim is None or any([i is None for i in dim]):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(name, "input size not accessible")
        # verify if output size % input size = 0 for all dim
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "output size that are not factor of input size"
            )
        k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
        # call max_poolxd_with_indices to get indices in the output
        if type == "MaxPool":
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        output = g.op(type, input, kernel_shape_i=tuple_fn(k), strides_i=tuple_fn(k))
        return output

    return symbolic_fn


adaptive_avg_pool1d = _adaptive_pool(
    "adaptive_avg_pool1d", "AveragePool", torch.nn.modules.utils._single
)
adaptive_avg_pool2d = _adaptive_pool(
    "adaptive_avg_pool2d", "AveragePool", torch.nn.modules.utils._pair
)
adaptive_avg_pool3d = _adaptive_pool(
    "adaptive_avg_pool3d", "AveragePool", torch.nn.modules.utils._triple
)

adaptive_max_pool1d = _adaptive_pool(
    "adaptive_max_pool1d",
    "MaxPool",
    torch.nn.modules.utils._single,
    max_pool1d_with_indices,
)
adaptive_max_pool2d = _adaptive_pool(
    "adaptive_max_pool2d",
    "MaxPool",
    torch.nn.modules.utils._pair,
    max_pool2d_with_indices,
)
adaptive_max_pool3d = _adaptive_pool(
    "adaptive_max_pool3d",
    "MaxPool",
    torch.nn.modules.utils._triple,
    max_pool3d_with_indices,
)


# Generate paddings in ONNX order based on pad in pytorch.
# Args:
#     dim: the dimension of the tensor.
#     pad: the paddings in pytorch.
#          The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ...
def _prepare_onnx_paddings(dim, pad):
    assert isinstance(dim, int)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    return paddings


def _convert_padding_node(padding):
    padding = symbolic_helper._maybe_get_const(padding, "is")
    if symbolic_helper._is_value(padding) and symbolic_helper._is_packed_list(padding):
        input_list = symbolic_helper._unpack_list(padding)
        try:
            padding = [
                symbolic_helper._get_const(v, "i", "padding") for v in input_list
            ]
        except Exception:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "Pad", 9, 11, "The sizes of the padding must be constant"
            )
    return padding


def constant_pad_nd(g, input, padding, value):
    mode = "constant"
    try:
        value = symbolic_helper._get_const(value, "f", "value")
    except Exception:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Pad", 9, 11, "The value for the padding must be constant"
        )

    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, value_f=value, opset_before=11
    )


def _pad_circular(g, input, pad):
    padding = _convert_padding_node(pad)
    assert len(padding) % 2 == 0
    ndim = len(padding) // 2

    cur = input
    for idx in range(ndim):
        pad_l = padding[-(2 * idx + 1)]
        pad_r = padding[-(2 * idx + 2)]

        tensors = []
        if pad_l > 0:
            left = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[-(pad_l + 1)], ends=[-1]
            )
            tensors.append(left)

        if pad_l < 0 or pad_r < 0:
            middle = symbolic_helper._slice_helper(
                g,
                cur,
                axes=[2 + idx],
                starts=[max(0, -pad_l)],
                ends=[-(1 + max(0, -pad_r))],
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


def reflection_pad(g, input, padding):
    mode = "reflect"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


def replication_pad(g, input, padding):
    mode = "edge"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def pad(g, input, pad, mode, value):
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
        raise RuntimeError(f"Unrecognized padding mode {mode}")


def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True")
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


upsample_nearest1d = _interpolate("upsample_nearest1d", 3, "nearest")
upsample_nearest2d = _interpolate("upsample_nearest2d", 4, "nearest")
upsample_nearest3d = _interpolate("upsample_nearest3d", 5, "nearest")
upsample_linear1d = _interpolate("upsample_linear1d", 3, "linear")
upsample_bilinear2d = _interpolate("upsample_bilinear2d", 4, "linear")
upsample_trilinear3d = _interpolate("upsample_trilinear3d", 5, "linear")


def __interpolate(
    g, input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Upsample", input, scales, mode_s=mode)


def bitwise_not(g, inp):
    if inp.type().scalarType() != "Bool":
        raise NotImplementedError(
            "ONNX export does NOT support exporting bitwise Not "
            + "for non-boolean input values"
        )
    return g.op("Not", inp)


def wrap_logical_op_with_cast_to(to_type):
    def decorator(fn):
        def wrap_with_cast(g, input, other):
            return g.op(
                "Cast",
                fn(g, input, other),
                to_i=symbolic_helper.cast_pytorch_to_onnx[to_type],
            )

        return wrap_with_cast

    return decorator


def wrap_logical_op_with_cast_to_and_from(to_type):
    def decorator(fn):
        def wrap_with_cast(g, input, other):
            to_cast_func = globals()[f"_cast_{to_type}"]
            from_cast_func = wrap_logical_op_with_cast_to(input.type().scalarType())(fn)
            return from_cast_func(
                g, to_cast_func(g, input, False), to_cast_func(g, other, False)
            )

        return wrap_with_cast

    return decorator


def wrap_logical_op_with_negation(func):
    def wrap_with_not(g, input, other):
        return g.op("Not", func(g, input, other))

    return wrap_with_not


def __not_(g, self):
    if self.type().scalarType() != "Bool":
        raise NotImplementedError(
            "ONNX export does NOT support exporting bitwise Not "
            + "for non-boolean input values"
        )
    return g.op("Not", self)


def eq(g, self, other):
    if isinstance(self.type(), _C.DeviceObjType) and isinstance(
        other.type(), _C.DeviceObjType
    ):
        # ONNX doesn't have devices, so consider them all to be equal.
        # The no-op check for equality will get constant-folded.
        return g.op("Constant", value_t=torch.tensor(True, dtype=torch.bool))
    return g.op("Equal", self, other)


@wrap_logical_op_with_negation
def ne(g, self, other):
    return eq(g, self, other)


def gt(g, input, other):
    return gt_impl(g, input, other)


def gt_impl(g, input, other):
    if (
        input.type().scalarType() is not None
        and input.type().scalarType() == "Bool"
        and other.type().scalarType() is not None
        and other.type().scalarType() == "Bool"
    ):
        input = g.op("Cast", input, to_i=symbolic_helper.cast_pytorch_to_onnx["Int"])
        other = g.op("Cast", other, to_i=symbolic_helper.cast_pytorch_to_onnx["Int"])
    return g.op("Greater", input, other)


def lt(g, input, other):
    return lt_impl(g, input, other)


def lt_impl(g, input, other):
    if (
        input.type().scalarType() is not None
        and input.type().scalarType() == "Bool"
        and other.type().scalarType() is not None
        and other.type().scalarType() == "Bool"
    ):
        input = g.op("Cast", input, to_i=symbolic_helper.cast_pytorch_to_onnx["Int"])
        other = g.op("Cast", other, to_i=symbolic_helper.cast_pytorch_to_onnx["Int"])
    return g.op("Less", input, other)


@wrap_logical_op_with_negation
def ge(g, input, other):
    return lt_impl(g, input, other)


@wrap_logical_op_with_negation
def le(g, input, other):
    return gt_impl(g, input, other)


def __and_(g, input, other):
    if input.type().scalarType() == "Bool" and other.type().scalarType() == "Bool":
        return g.op("And", input, other)
    else:
        raise NotImplementedError(
            "ONNX export does NOT support exporting bitwise AND "
            + "for non-boolean input values"
        )


def __or_(g, input, other):
    if input.type().scalarType() == "Bool" and other.type().scalarType() == "Bool":
        return g.op("Or", input, other)
    else:
        raise NotImplementedError(
            "ONNX export does NOT support exporting bitwise OR "
            + "for non-boolean input values"
        )


def __xor_(g, input, other):
    if input.type().scalarType() == "Bool" and other.type().scalarType() == "Bool":
        return g.op("Xor", input, other)
    else:
        raise NotImplementedError(
            "ONNX export does NOT support exporting bitwise XOR "
            + "for non-boolean input values"
        )


@wrap_logical_op_with_cast_to_and_from("Bool")
def logical_and(g, input, other):
    return g.op("And", input, other)


@wrap_logical_op_with_cast_to_and_from("Bool")
def logical_or(g, input, other):
    return g.op("Or", input, other)


@wrap_logical_op_with_cast_to_and_from("Bool")
def logical_xor(g, input, other):
    return g.op("Xor", input, other)


def __rshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op(
            "Cast",
            other,
            to_i=symbolic_helper.cast_pytorch_to_onnx[self.type().scalarType()],
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=symbolic_helper.cast_pytorch_to_onnx[self.type().scalarType()],
    )
    rshift = g.op("Div", self, two_pow)
    return rshift


def __lshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op(
            "Cast",
            other,
            to_i=symbolic_helper.cast_pytorch_to_onnx[self.type().scalarType()],
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=symbolic_helper.cast_pytorch_to_onnx[self.type().scalarType()],
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


@symbolic_helper.parse_args("v", "v", "v", "i")
def where(g, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if condition.type().scalarType() != "Bool":
        condition = g.op(
            "Cast", condition, to_i=symbolic_helper.cast_pytorch_to_onnx["Bool"]
        )
    if self is None:
        condition = nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    return g.op("Where", condition, self, other)


@symbolic_helper.parse_args("v", "i", "none")
def log_softmax(g, input, dim, dtype=None):
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
            "Cast", return_op, to_i=symbolic_helper.scalar_type_to_onnx[parsed_dtype]
        )
    if is_transpose_required:
        return_op = g.op("Transpose", return_op, perm_i=axes)
    return return_op


@symbolic_helper.parse_args(
    "v", "v", "v", "is", "is", "is", "i", "is", "i", "i", "i", "i", "i"
)
def _convolution(
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
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        kernel_shape = None

    if kernel_shape is None or any([i is None for i in kernel_shape]):
        raise RuntimeError(
            "Unsupported: ONNX export of convolution for kernel " "of unknown shape."
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i")
def conv1d(g, input, weight, bias, stride, padding, dilation, groups):
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i")
def conv2d(g, input, weight, bias, stride, padding, dilation, groups):
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i")
def conv3d(g, input, weight, bias, stride, padding, dilation, groups):
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose1d(
    g, input, weight, bias, stride, padding, output_padding, groups, dilation
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose2d(
    g, input, weight, bias, stride, padding, output_padding, groups, dilation
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


@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose3d(
    g, input, weight, bias, stride, padding, output_padding, groups, dilation
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


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g,
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


@symbolic_helper.parse_args("v", "is", "v", "v", "f", "i")
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
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

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = symbolic_helper._generate_wrapped_number(g, 2.0)
    eps_cst = symbolic_helper._generate_wrapped_number(g, eps)

    mean = g.op("ReduceMean", input, axes_i=axes)
    numerator = sub(g, input, mean)
    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, add(g, variance, eps_cst))

    layer_norm = g.op("Div", numerator, denominator)

    if not (weight is None or symbolic_helper._is_none(weight)):
        layer_norm = mul(g, layer_norm, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        layer_norm = add(g, layer_norm, bias)

    return layer_norm


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def instance_norm(
    g,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats,
    momentum,
    eps,
    cudnn_enabled,
):
    symbolic_helper.check_training_mode(use_input_stats, "instance_norm")
    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if weight is None or symbolic_helper._is_none(weight):
        if channel_size is None:
            raise RuntimeError(
                "Unsupported: ONNX export of instance_norm for unknown " "channel size."
            )
        weight_value = torch.tensor([1.0] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor"
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or symbolic_helper._is_none(bias):
        if channel_size is None:
            raise RuntimeError(
                "Unsupported: ONNX export of instance_norm for unknown " "channel size."
            )
        bias_value = torch.tensor([0.0] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor"
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
            raise RuntimeError(
                "Unsupported: ONNX export of instance_norm training for unknown "
                "batch size."
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


@symbolic_helper.parse_args("v", "i", "i", "i")
def unfold(g, input, dimension, size, step):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("unfold", input, dimension_i=dimension, size_i=size, step_i=step)
    sizes = symbolic_helper._get_tensor_sizes(input)
    try:
        sizedim = sizes[dimension]
    except Exception:
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
        return symbolic_helper._unimplemented("Unfold", "input size not accessible")


@symbolic_helper.parse_args("v", "t", "t", "t")
def elu(g, input, alpha, scale, input_scale):
    if scale and scale != 1.0:
        return symbolic_helper._unimplemented("scale", "does not support scale in Elu")
    if input_scale and input_scale != 1.0:
        return symbolic_helper._unimplemented(
            "input_scale", "does not support input_scale in Elu"
        )
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=symbolic_helper._scalar(alpha))


def selu(g, input):
    return g.op("Selu", input)


@symbolic_helper.parse_args("v", "i", "v")
def index_select(g, self, dim, index):
    # In case of a scalar index, index_select returns a tensor with the same rank as the input.
    # To match this behavior in ONNX, we make index a 1D tensor so that the following gather
    # also produces a tensor with the same rank as the input.
    return symbolic_helper._select_helper(g, self, dim, index)


def index_put(g, self, indices_list_value, values, accumulate):
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
        else:
            return values
    else:
        symbolic_helper._onnx_opset_unsupported("index_put", 9, 11)


def index_fill(g, self, dim, index, value):
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
    value = symbolic_helper._if_scalar_type_as(g, value, self)
    expanded_value = expand(g, value, expanded_index_shape, None)

    return scatter(g, self, dim, expanded_index, expanded_value)


def index_copy(g, self, dim, index, source):
    dim_value = symbolic_helper._parse_arg(dim, "i")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index_copy", self, index, source, dim_i=dim_value)
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@symbolic_helper.parse_args("v", "v", "b", "b")
def bucketize(g, self, boundaries, out_int32=False, right=False):
    out_type = _C_onnx.TensorProtoDataType.INT64
    if out_int32:
        out_type = _C_onnx.TensorProtoDataType.INT32
    # A tensor expanded_boundaries is created such that it
    # contains a copy of boundaries for each element of self.
    new_shape = g.op("Concat", g.op("Shape", boundaries), g.op("Shape", self), axis_i=0)
    # Unsqueeze step is performed to respect ONNX's numpy style broadcasting for comparison ops
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    unsqueeze_axes = list(range(1, symbolic_helper._get_tensor_rank(self) + 1))
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


def type_as(g, self, other):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    other_dtype = symbolic_helper._try_get_scalar_type(other)
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    if other_dtype is not None:
        return g.op(
            "Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx[other_dtype]
        )
    else:
        if symbolic_helper.is_caffe2_aten_fallback():
            # We don't know the type of other, bail by emitting ATen
            return g.at("type_as", self, other)
        else:
            raise RuntimeError(
                "Unsupported: ONNX export of type_as for tensor "
                "of unknown dtype. Please check if the dtype of the "
                "parameter passed to the type_as function is correct."
            )


@symbolic_helper.parse_args("v", "v", "i", "f")
def cosine_similarity(g, x1, x2, dim, eps):
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


def pairwise_distance(g, input1, input2, p, eps, keepdim):
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


# ignore clone operators that are inserted by PyTorch autograd
def clone(g, input, unused_memory_format):
    return input


def abs(g, self):
    return g.op("Abs", self)


def log(g, self):
    return g.op("Log", self)


def log1p(g, self):
    return log(
        g, add(g, symbolic_helper._if_scalar_type_as(g, torch.ones(1), self), self)
    )


def log10(g, self):
    _ln10 = 2.30258509299404568401
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor([_ln10])))


def pow(g, self, exponent):
    f_dtype = self_dtype = self.type().scalarType()
    if not symbolic_helper._is_fp(self):
        f_dtype = "Float"
        self = g.op("Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx[f_dtype])
    if not symbolic_helper._is_fp(exponent):
        exponent = g.op(
            "Cast", exponent, to_i=symbolic_helper.cast_pytorch_to_onnx[f_dtype]
        )
    pow = g.op("Pow", self, exponent)
    return pow


def clamp(g, self, min, max):
    # min or max may be None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax
    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if symbolic_helper._is_constant(min) and symbolic_helper._is_constant(max):
            return op_with_optional_float_cast(
                g,
                "Clip",
                self,
                min_f=symbolic_helper._parse_arg(min, "f"),
                max_f=symbolic_helper._parse_arg(max, "f"),
                opset_before=12,
            )
        else:
            return clamp_max(g, clamp_min(g, self, min), max)


@symbolic_helper.parse_args("v", "v")
def clamp_min(g, self, min):
    if symbolic_helper._is_constant(min):
        return op_with_optional_float_cast(
            g, "Clip", self, min_f=symbolic_helper._parse_arg(min, "f"), opset_before=12
        )
    else:
        dtype = self.type().scalarType()
        min = g.op("Cast", min, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype])
        return op_with_optional_float_cast(g, "Max", self, min, opset_before=12)


@symbolic_helper.parse_args("v", "v")
def clamp_max(g, self, max):
    if symbolic_helper._is_constant(max):
        return op_with_optional_float_cast(
            g, "Clip", self, max_f=symbolic_helper._parse_arg(max, "f"), opset_before=12
        )
    else:
        dtype = self.type().scalarType()
        max = g.op("Cast", max, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype])
        return op_with_optional_float_cast(g, "Min", self, max, opset_before=12)


# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
def max(g, self, dim_or_y=None, keepdim=None):
    # torch.max(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    # torch.max(input, other)
    if keepdim is None:
        return op_with_optional_float_cast(g, "Max", self, dim_or_y, opset_before=12)
    # torch.max(input, dim, keepdim)
    else:
        dim = symbolic_helper._get_const(dim_or_y, "i", "dim")
        keepdim = symbolic_helper._get_const(keepdim, "i", "keepdim")
        max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op("ArgMax", self, axis_i=dim, keepdims_i=keepdim)
        return max, indices


def maximum(g, input, other):
    return max(g, input, dim_or_y=other)


def min(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    # torch.min(input, other)
    if keepdim is None:
        return op_with_optional_float_cast(g, "Min", self, dim_or_y, opset_before=12)
    # torch.min(input, dim, keepdim)
    else:
        dim = symbolic_helper._get_const(dim_or_y, "i", "dim")
        keepdim = symbolic_helper._get_const(keepdim, "i", "keepdim")
        min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op("ArgMin", self, axis_i=dim, keepdims_i=keepdim)
        return min, indices


def minimum(g, input, other):
    return min(g, input, dim_or_y=other)


@symbolic_helper.parse_args("v", "is", "i")
def amax(g, self, dim, keepdim):
    return g.op("ReduceMax", self, axes_i=dim, keepdims_i=keepdim)


@symbolic_helper.parse_args("v", "is", "i")
def amin(g, self, dim, keepdim):
    return g.op("ReduceMin", self, axes_i=dim, keepdims_i=keepdim)


@symbolic_helper.parse_args("v", "v", "i")
def aminmax(g, self, dim, keepdim):
    reduce_kwargs = {"keepdims_i": keepdim}
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        reduce_kwargs["axes_i"] = [dim]

    return g.op("ReduceMin", self, **reduce_kwargs), g.op(
        "ReduceMax", self, **reduce_kwargs
    )


def exp(g, self):
    return g.op("Exp", self)


@symbolic_helper.parse_args("v", "f", "i")
def dropout(g, input, p, train):
    symbolic_helper.check_training_mode(train, "dropout")
    # if train is False, dropout is no-op
    if not train:
        return input
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


def _unsupported_dropout(name):
    @symbolic_helper.parse_args("v", "f", "i")
    def feature_dropout(g, input, p, train):
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        if train:
            return symbolic_helper._unimplemented(name, "training mode")
        return input

    return feature_dropout


feature_dropout = _unsupported_dropout("feature_dropout")
alpha_dropout = _unsupported_dropout("alpha_dropout")
feature_alpha_dropout = _unsupported_dropout("feature_alpha_dropout")

# See Note [Export inplace]
dropout_ = dropout
feature_dropout_ = feature_dropout
alpha_dropout_ = alpha_dropout
feature_alpha_dropout_ = feature_alpha_dropout


@symbolic_helper.parse_args("v", "t", "is", "i")
def norm(g, self, p, dim, keepdim):
    if p == 1:
        f = _reduce_op_symbolic("ReduceL1")
    elif p == 2:
        f = _reduce_op_symbolic("ReduceL2")
    else:
        raise RuntimeError("ONNX export only p-norms with p of 1 or 2")
    return f(g, self, dim=dim, keepdim=keepdim)


@symbolic_helper.parse_args("v", "v", "v", "i")
def conv_tbc(g, input, weight, bias, pad):
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


@symbolic_helper.parse_args("v", "i", "i")
def _unique(g, input, sorted, return_inverse):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "_unique",
            input,
            sorted_i=sorted,
            return_inverse_i=return_inverse,
            outputs=2,
        )
    else:
        return symbolic_helper._onnx_unsupported("_unique")


@symbolic_helper.parse_args("v", "i", "i", "i")
def _unique2(g, input, sorted, return_inverse, return_counts):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "_unique2",
            input,
            sorted_i=sorted,
            return_inverse_i=return_inverse,
            return_counts_i=return_counts,
            outputs=3,
        )
    else:
        symbolic_helper._onnx_opset_unsupported("_unique2", 9, 11)


# TODO(justinchuby): Clean up this function generation magic by defining the functions
# explicitly.
for k, v in symbolic_helper.cast_pytorch_to_onnx.items():  # type: ignore[has-type]
    name = f"_cast_{k}"
    globals()[name] = symbolic_helper.parse_args("v", "i")(
        functools.partial(symbolic_helper._cast_func_template, v)
    )


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty_like(
    g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None
):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


def new_empty(g, self, sizes, dtype, layout, device, pin_memory=False):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
    return empty(g, sizes, dtype, layout, device, pin_memory)


def scalar_tensor(g, scalar, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    scalar = g.op("Cast", scalar, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    return scalar


def tensor(g, data, dtype=None, device=None, requires_grad=False):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if symbolic_helper._is_packed_list(data):
        if dtype is None:
            dtype = symbolic_helper._unpack_list(data)[0].type().scalarType()
            dtype = symbolic_helper.scalar_type_to_onnx.index(
                symbolic_helper.cast_pytorch_to_onnx[dtype]
            )
        input_list = list()
        for t in symbolic_helper._unpack_list(data):
            shape_reference = g.op("Constant", value_t=torch.LongTensor([1]))
            t = symbolic_helper._reshape_helper(g, t, shape_reference)
            t = g.op("Cast", t, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
            input_list.append(t)
        return g.op("Concat", *input_list, axis_i=0)
    else:
        if dtype is None:
            dtype = data.type().scalarType()
            dtype = symbolic_helper.scalar_type_to_onnx.index(
                symbolic_helper.cast_pytorch_to_onnx[dtype]
            )
        if symbolic_helper._is_list(data) and (
            symbolic_helper._is_tensor_list(data)
            or symbolic_helper._is_scalar_list(data)
        ):
            data = g.op("ConcatFromSequence", data, axis_i=0, new_axis_i=1)
    return g.op("Cast", data, to_i=symbolic_helper.scalar_type_to_onnx[dtype])


def as_tensor(g, data, dtype=None, device=None):
    return tensor(g, data, dtype, device)


@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device, layout and pin_memory in ONNX, so we ignore it
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor(
            [0], dtype=symbolic_helper.scalar_type_to_pytorch_type[dtype]
        ),
    )


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(
    g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None
):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor(
            [0], dtype=symbolic_helper.scalar_type_to_pytorch_type[dtype]
        ),
    )


def new_zeros(g, self, sizes, dtype, layout, device, pin_memory=False):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor(
            [1], dtype=symbolic_helper.scalar_type_to_pytorch_type[dtype]
        ),
    )


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(
    g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None
):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor(
            [1], dtype=symbolic_helper.scalar_type_to_pytorch_type[dtype]
        ),
    )


def new_ones(g, self, sizes, dtype, layout, device, pin_memory=False):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
    return ones(g, sizes, dtype, layout, device, pin_memory)


def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    const_value = symbolic_helper._maybe_get_const(value, "t")
    if symbolic_helper._is_value(const_value):
        dtype = symbolic_helper.ScalarType.FLOAT if dtype is None else dtype
        tmp = zeros(g, sizes, dtype, layout, device)
        return add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        dtype = symbolic_helper.ScalarType.FLOAT if dtype is None else dtype
        sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
        if isinstance(sizes_, list) and len(sizes_) == 0:
            sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
        return g.op(
            "ConstantOfShape",
            sizes,
            value_t=const_value.view(1).to(
                symbolic_helper.scalar_type_to_pytorch_type[dtype]
            ),
        )


def full_like(
    g,
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
    dtype = symbolic_helper.ScalarType.FLOAT if dtype is None else dtype
    if symbolic_helper._is_value(fill_value):
        tmp = zeros_like(g, input, dtype, layout, device)
        fill_value = g.op(
            "Cast", fill_value, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
        )
        return add(g, tmp, fill_value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        shape = g.op("Shape", input)
        return g.op(
            "ConstantOfShape",
            shape,
            value_t=torch.tensor([fill_value]).to(
                symbolic_helper.scalar_type_to_pytorch_type[dtype]
            ),
        )


def new_full(g, self, size, fill_value, dtype, layout, device, pin_memory=False):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
    return full(g, size, fill_value, dtype, layout, device, pin_memory)


def eye(g, *args):
    if len(args) == 5:
        # aten::eye(n, dtype, layout, device, pin_memory)
        n, dtype, layout, device, pin_memory = args
        dim_size = symbolic_helper._unsqueeze_helper(g, n, [0])
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    elif len(args) == 6:
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
    else:
        raise NotImplementedError("Unknown aten::eye signature")


def slice(g, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        dim, start, end, step = args
        step = symbolic_helper._parse_arg(step, "i")
        if step != 1:
            raise RuntimeError("step!=1 is currently not supported")
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
                raise RuntimeError(
                    "Unsupported: ONNX export of Slice with dynamic inputs. DynamicSlice "
                    "is a deprecated experimental op. Please use statically allocated "
                    "variables or export to a higher opset version."
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
                9223372036854775807
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
            9223372036854775807 if is_end_none else symbolic_helper._parse_arg(end, "i")
        )
        return symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[start], ends=[end]
        )
    else:
        raise NotImplementedError("Unknown aten::slice signature")


@symbolic_helper.parse_args("v", "f", "f")
def hardtanh(g, self, min_val, max_val):
    return op_with_optional_float_cast(
        g, "Clip", self, min_f=min_val, max_f=max_val, opset_before=12
    )


@symbolic_helper.parse_args("v")
def hardswish(g, self):
    hs = hardsigmoid(g, self)
    return g.op("Mul", self, hs)


# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qhardsigmoid.cpp
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
@symbolic_helper.parse_args("v")
def hardsigmoid(g, self):
    # Set alpha_f to 1 / 6 to make op equivalent to PyTorch's definition of Hardsigmoid.
    # See https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    return g.op("HardSigmoid", self, alpha_f=1 / 6)


@symbolic_helper.parse_args("v")
def tanhshrink(g, self):
    return g.op("Sub", self, tanh(g, self))


@symbolic_helper.parse_args("v", "f")
def hardshrink(g, self, lambd):
    lambd_op = g.op("Constant", value_t=torch.FloatTensor([lambd]))
    cond = logical_or(g, gt(g, self, lambd_op), lt(g, self, neg(g, lambd_op)))
    return g.op("Where", cond, self, g.op("Constant", value_t=torch.FloatTensor([0])))


@symbolic_helper.parse_args("v", "f")
def softshrink(g, self, lambd):
    lambd_op = g.op("Constant", value_t=torch.FloatTensor([lambd]))
    gt_cond = gt(g, self, lambd_op)
    gt_out = g.op(
        "Where",
        gt_cond,
        sub(g, self, lambd_op),
        g.op("Constant", value_t=torch.FloatTensor([0])),
    )
    lt_cond = lt(g, self, neg(g, lambd_op))
    lt_out = g.op(
        "Where",
        lt_cond,
        add(g, self, lambd_op),
        g.op("Constant", value_t=torch.FloatTensor([0])),
    )
    return add(g, gt_out, lt_out)


def alias(g, self):
    return self


@symbolic_helper.parse_args("v", "i")
def unsqueeze(g, self, dim):
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
                "unsqueeze", "negative axis with unknown input rank"
            )

    return symbolic_helper._unsqueeze_helper(g, self, axes_i=[dim])


@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g, self, dim, decending, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Sort", "Out parameter is not supported for sort"
        )
    self_sizes = symbolic_helper._get_tensor_sizes(self)
    try:
        dim_size = self_sizes[dim]
    except Exception:
        dim_size = None

    if dim_size is None:
        return symbolic_helper._unimplemented("Sort", "input size not accessible")

    return g.op("TopK", self, k_i=dim_size, axis_i=dim, outputs=2)


def numel(g, self):
    shape = g.op("Shape", self)
    return g.op("ReduceProd", shape, keepdims_i=0)


@symbolic_helper.parse_args("v", "i", "i", "i", "i", "none")
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "TopK", "Out parameter is not supported for topk"
        )
    if not largest:
        symbolic_helper._unimplemented("TopK", "Ascending TopK is not supported")

    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


def to(g, self, *args):
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
            tval = args[0].node()["value"]
            if isinstance(tval, torch.Tensor):
                if len(tval.shape) == 0:
                    tval = tval.item()
                    dtype = int(tval)
                else:
                    dtype = tval

        if symbolic_helper._is_value(dtype) or isinstance(dtype, torch.Tensor):
            # aten::to(Tensor, Tensor, bool, bool, memory_format)
            dtype = args[0].type().scalarType()
            return g.op("Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype])
        else:
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            # memory_format is ignored
            return g.op("Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        dtype = symbolic_helper._get_const(args[1], "i", "dtype")
        # memory_format is ignored
        return g.op("Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) -> Tensor
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) -> Tensor
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    else:
        return symbolic_helper._onnx_unsupported("Unknown aten::to signature")


def repeat(g, self, repeats):
    dtype = symbolic_helper.ScalarType.INT64
    shape_ = ones_like(g, repeats, dtype)
    self = g.op("Expand", self, shape_)
    return g.op("Tile", self, repeats)


def repeat_interleave(g, self, repeats, dim=None, output_size=None):
    input = self
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if symbolic_helper._is_none(dim):
        input = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = 0
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)

    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    input_sizes = symbolic_helper._get_tensor_sizes(input)
    if repeats_dim is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats rank."
        )
    if repeats_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size."
        )
    if input_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size."
        )

    input_sizes_temp = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            input_sizes[idx], input_sizes_temp[idx] = 0, -1

    # Cases where repeats is an int or single value tensor
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        if not symbolic_helper._is_tensor(repeats):
            repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        else:
            reps = input_sizes[dim]
            repeats = expand(
                g, repeats, g.op("Constant", value_t=torch.tensor([reps])), None
            )

    # Cases where repeats is a 1 dim Tensor
    elif repeats_dim == 1:
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        if repeats_sizes[0] is None:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave", 9, 13, "Unsupported for cases with dynamic repeats"
            )
        assert (
            repeats_sizes[0] == input_sizes[dim]
        ), "repeats must have the same size as input along dim"
        reps = repeats_sizes[0]
    else:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    final_splits = list()
    r_splits = symbolic_helper._repeat_interleave_split_helper(g, repeats, reps, 0)
    i_splits = symbolic_helper._repeat_interleave_split_helper(g, input, reps, dim)
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


@symbolic_helper.parse_args("v", "i")
def pixel_shuffle(g, self, upscale_factor):
    dims = symbolic_helper._get_tensor_sizes(self)
    if len(dims) != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
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


@symbolic_helper.parse_args("v", "i")
def pixel_unshuffle(g, self, downscale_factor):
    dims = symbolic_helper._get_tensor_sizes(self)
    if len(dims) != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
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


def _generic_rnn(
    g,
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
        return symbolic_helper._unimplemented("LSTM", "LSTMs with projections")
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
            "RNN/GRU/LSTM", "dropout in training mode"
        )

    if variant.startswith("RNN"):
        nonlinearity = variantToOnnxActivationMap[variant[4:].lower()]
        variant = "RNN"

    w_hh = all_weights[1]
    hidden_size = symbolic_helper._get_tensor_dim_size(w_hh, 1)
    if hidden_size is None:
        return symbolic_helper._unimplemented("RNN/GRU/LSTM", "unknown hidden size")

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

    def reform_weights(g, w, n, intervals):
        slices = [
            symbolic_helper._slice_helper(g, w, axes=[0], starts=[x * n], ends=[y * n])
            for x, y in intervals
        ]
        return g.op("Concat", *slices, axis_i=0)

    def transform_weights_no_bias(layer_index):
        weights = layer_weights[layer_index]
        if variant == "RNN":
            weight_ih, weight_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            weight_ih, weight_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh)
        )

    def transform_weights(layer_index):
        weights = layer_weights[layer_index]
        if variant == "RNN":
            weight_ih, weight_hh, bias_ih, bias_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            weight_ih, weight_hh, bias_ih, bias_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        bias_concat = g.op("Concat", bias_ih, bias_hh, axis_i=0)
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0])
            for x in (weight_ih, weight_hh, bias_concat)
        )

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

        inputs.append(retrieve_state(h0, *state_indices))
        if variant == "LSTM":
            inputs.append(retrieve_state(c0, *state_indices))

        extra_kwargs = {} if unidirectional else {"direction_s": "bidirectional"}
        if variant == "RNN":
            if bidirectional:
                activation = [nonlinearity, nonlinearity]
            else:
                activation = [nonlinearity]

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

        h_outs.append(h_out)
        if variant == "LSTM":
            c_outs.append(c_out)
    if batch_first:
        # seq, batch, num_directions * hidden_size -> batch, seq, num_directions * hidden_size
        prev_output = g.op("Transpose", prev_output, perm_i=[1, 0, 2])
    h_outs = h_out if num_layers == 1 else g.op("Concat", *h_outs, axis_i=0)
    if variant == "RNN" or variant == "GRU":
        return prev_output, h_outs
    elif variant == "LSTM":
        c_outs = c_out if num_layers == 1 else g.op("Concat", *c_outs, axis_i=0)
        return prev_output, h_outs, c_outs


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "f", "i", "i", "i")
def _lstm_full(
    g,
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
def _lstm_packed(
    g,
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


def lstm(g, *args):
    if symbolic_helper._is_tensor_list(args[3]):
        return _lstm_packed(g, *args)
    else:
        return _lstm_full(g, *args)


def lstm_cell(g, self, hidden, w_ih, w_hh, b_ih, b_hh):
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


def _one_hidden_rnn(kind):
    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "f", "i", "i", "i")
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


gru = _one_hidden_rnn("GRU")
rnn_tanh = _one_hidden_rnn("RNN_TANH")
rnn_relu = _one_hidden_rnn("RNN_RELU")


@symbolic_helper.parse_args("v", "i")
def _dim_arange(g, like, dim):
    like_shape = g.op("Shape", like)
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.op("_caffe2::Range", stop)
    else:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        return arange(g, stop, 4, None, None, None)


def detach(g, input):
    # Erase aten::detach nodes because ONNX is inference only
    return input


@symbolic_helper.parse_args("v", "i")
def contiguous(g, input, memory_format):
    if memory_format > 2:  # allower values are any, preserve and contiguous_format
        raise RuntimeError("onnx memory_format support is not implemented")
    return input


@symbolic_helper.parse_args("v", "v", "i")
def _pack_padded_sequence(g, input, lengths, batch_first):
    # Currently there is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.
    if batch_first:
        input = g.op("Transpose", input, perm_i=[1, 0, 2])
    if not lengths.type().isSubtypeOf(torch._C.TensorType.get()):
        raise RuntimeError("Lengths must be a Tensor for ONNX export")
    # We know it's a TensorType so this check is now safe.
    # It's really only necessary because those operators expand to something that
    # only works with int32 types in Caffe2...
    if lengths.type().scalarType() != "Int":
        lengths = _cast_Int(g, lengths, False)  # type: ignore[name-defined]
    return g.op("prim::PackPadded", input, lengths, outputs=2)


@symbolic_helper.parse_args("v", "v", "i", "t", "v")
def _pad_packed_sequence(
    g, data, batch_sizes, batch_first, padding_value, total_length
):
    # Ignore total_length as it is not supported in _symbolic_pad_packed_sequence
    # It is only useful/used when training using data_parallel model, so
    # It shouldn't be relevant for ONNX anyway
    data, lengths = g.op("prim::PadPacked", data, batch_sizes, outputs=2)
    if batch_first:
        data = g.op("Transpose", data, perm_i=[1, 0, 2])
    return data, lengths


def randn(g, shapes, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor(
                [0], dtype=symbolic_helper.scalar_type_to_pytorch_type[6]
            ),
        )
        return g.op(
            "RandomNormalLike",
            shape_const,
            dtype_i=symbolic_helper.scalar_type_to_onnx[dtype],
        )
    return g.op(
        "RandomNormal",
        shape_i=shape,
        dtype_i=symbolic_helper.scalar_type_to_onnx[dtype],
    )


def rand(g, shapes, dtype, *options):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor(
                [0], dtype=symbolic_helper.scalar_type_to_pytorch_type[6]
            ),
        )
        return g.op(
            "RandomUniformLike",
            shape_const,
            dtype_i=symbolic_helper.scalar_type_to_onnx[dtype],
        )
    return g.op(
        "RandomUniform",
        shape_i=shape,
        dtype_i=symbolic_helper.scalar_type_to_onnx[dtype],
    )


def randn_like(
    g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None
):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    return g.op(
        "RandomNormalLike", self, dtype_i=symbolic_helper.scalar_type_to_onnx[dtype]
    )


def rand_like(
    g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None
):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    return g.op(
        "RandomUniformLike", self, dtype_i=symbolic_helper.scalar_type_to_onnx[dtype]
    )


@symbolic_helper.parse_args("v", "f", "f", "i", "none")
def rrelu(g, input, lower, upper, training, generator):
    p = g.op("RandomUniformLike", input, high_f=upper, low_f=lower)
    return g.op("PRelu", input, p)


def bernoulli(g, input, generator=None, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli"
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli"
        )

    dtype = symbolic_helper._try_get_scalar_type(input)
    if dtype is None:
        return symbolic_helper._unimplemented("Bernoulli", "input dtype not accessible")
    p = g.op(
        "RandomUniformLike",
        input,
        high_f=1.0,
        low_f=0.0,
        dtype_i=symbolic_helper.cast_pytorch_to_onnx[dtype],
    )
    output = g.op("Less", p, input)
    return g.op("Cast", output, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype])


@symbolic_helper.parse_args("v")
def log_sigmoid(g, input):
    p = g.op("Sigmoid", input)
    return g.op("Log", p)


@symbolic_helper.parse_args("v")
def erf(g, input):
    return g.op("Erf", input)


@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
def flatten(g, input, start_dim, end_dim):
    dim = symbolic_helper._get_tensor_rank(input)
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )

    # TODO: remove this as onnx opset 11 spec allows negative axes
    if end_dim < 0:
        end_dim = dim + end_dim
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1 and end_dim == dim - 1:
        return g.op("Flatten", input, axis_i=start_dim)
    if start_dim == 0 and end_dim == dim - 2:
        return g.op("Flatten", input, axis_i=end_dim + 1)

    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)


@symbolic_helper.parse_args("v")
def nonzero(g, input):
    """Emitted from `torch.nonzero(x, as_tuple=False)`"""
    return t(g, g.op("NonZero", input))


# Emitted from `torch.nonzero(x, as_tuple=True)`
def nonzero_numpy(g, input, _outputs=None):
    return unbind(g, nonzero(g, input), 1, _outputs=_outputs)


@symbolic_helper.parse_args("v")
def isnan(g, input):
    output = g.op("IsNaN", input)
    return output


def _any(g, *args):
    # aten::any(Tensor self)
    if len(args) == 1:
        input = args[0]
        dim, keepdim = None, 0
    # aten::any(Tensor self, int dim, bool keepdim)
    else:
        input, dim, keepdim = args
        dim = [symbolic_helper._parse_arg(dim, "i")]
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
    input = _cast_Long(g, input, False)  # type: ignore[name-defined]
    input_sum = symbolic_helper._reducesum_helper(
        g, input, axes_i=dim, keepdims_i=keepdim
    )
    return gt(g, input_sum, g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)))


def _all(g, *args):
    input = g.op("Not", args[0])
    # aten::all(Tensor self)
    if len(args) == 1:
        return g.op("Not", _any(g, input))
    # aten::all(Tensor self, int dim, bool keepdim)
    else:
        return g.op("Not", _any(g, input, args[1], args[2]))


@symbolic_helper.parse_args("v", "i", "i", "i")
def narrow(g, input, dim, start, length):
    return symbolic_helper._slice_helper(
        g, input, axes=[dim], starts=[start], ends=[start + length]
    )


def argmax(g, input, dim, keepdim):
    if symbolic_helper._is_none(dim):
        flattened = symbolic_helper._reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        return g.op("ArgMax", flattened, axis_i=0, keepdims_i=False)
    else:
        dim = symbolic_helper._parse_arg(dim, "i")
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
        return g.op("ArgMax", input, axis_i=dim, keepdims_i=keepdim)


def argmin(g, input, dim, keepdim):
    if symbolic_helper._is_none(dim):
        flattened = symbolic_helper._reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        return g.op("ArgMin", flattened, axis_i=0, keepdims_i=False)
    else:
        dim = symbolic_helper._parse_arg(dim, "i")
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
        return g.op("ArgMin", input, axis_i=dim, keepdims_i=keepdim)


@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter(g, self, dim, index, src):
    src_type = src.type().scalarType()
    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("Scatter", self, index, src, axis_i=dim)
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if self.type().scalarType() != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=symbolic_helper.cast_pytorch_to_onnx[self.type().scalarType()],
            )
        return g.op("Scatter", self, index, expand_as(g, src, index), axis_i=dim)


@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter_add(g, self, dim, index, src):
    dtype = symbolic_helper._try_get_scalar_type(self)
    if dtype is None:
        return symbolic_helper._unimplemented(
            "scatter_add", "input dtype not accessible"
        )
    dtype = symbolic_helper.scalar_type_to_onnx.index(
        symbolic_helper.cast_pytorch_to_onnx[dtype]
    )
    dtype = symbolic_helper.scalar_type_to_pytorch_type[dtype]
    sizes = symbolic_helper._get_tensor_sizes(self, allow_nonstatic=False)
    if sizes:
        to_add = g.op("Constant", value_t=torch.zeros(sizes, dtype=dtype))
    else:
        dtype = symbolic_helper.scalar_type_to_pytorch_type.index(dtype)
        to_add = zeros_like(g, self, dtype)
    to_add = symbolic_helper._scatter_helper(g, to_add, dim, index, src)
    return add(g, self, to_add)


def log2(g, self):
    _ln2 = 0.693147180559945309
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor(_ln2)))


def is_floating_point(g, self):
    if symbolic_helper._is_fp(self):
        return g.op("Constant", value_t=torch.BoolTensor([1]))
    return g.op("Constant", value_t=torch.BoolTensor([0]))


def __is_(g, self, other):
    if symbolic_helper._is_none(other):
        if symbolic_helper._is_none(self):
            return g.op("Constant", value_t=torch.BoolTensor([1]))
        return g.op("Constant", value_t=torch.BoolTensor([0]))
    return eq(g, self, other)


@wrap_logical_op_with_negation
def __isnot_(g, self, other):
    return __is_(g, self, other)


def one_hot(g, self, num_classes):
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    # onnxruntime supports limited type combinations for OneHot.
    if num_classes.type().scalarType() in ("Byte", "Char", "Int", "Short"):
        num_classes = g.op(
            "Cast", num_classes, to_i=symbolic_helper.cast_pytorch_to_onnx["Long"]
        )
    return g.op("OneHot", self, num_classes, values, axis_i=-1)


@symbolic_helper.parse_args("v", "i", "v", "v")
def gather(g, self, dim, index, sparse_grad=False):
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True")
    # NOTE: This workaround is needed since GatherElement is only supported
    #       since opset 11, and Gather in ONNX is not the same as torch.gather.
    dtype = self.type().scalarType()
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    depth = size(g, self, g.op("Constant", value_t=torch.LongTensor([dim])))
    index = g.op(
        "Cast",
        g.op("OneHot", index, depth, values, axis_i=dim),
        to_i=symbolic_helper.cast_pytorch_to_onnx[dtype],
    )
    mul = g.op("Mul", symbolic_helper._unsqueeze_helper(g, self, [dim + 1]), index)
    return symbolic_helper._reducesum_helper(g, mul, axes_i=[dim], keepdims_i=0)


@symbolic_helper.parse_args("v", "is", "i", "i")
def _var_mean(g, input, dim, correction, keepdim):
    if dim is None:
        mean = g.op("ReduceMean", input, keepdims_i=0)
        t_mean = mean
        num_elements = numel(g, input)
    else:
        mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
        t_mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=1)
        redudced_dims = g.op("Shape", input)
        # dim could contain one or multiple dimensions
        redudced_dims = g.op(
            "Gather",
            redudced_dims,
            g.op("Constant", value_t=torch.tensor(dim)),
            axis_i=0,
        )
        num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
    sub_v = g.op("Sub", input, t_mean)
    sqr_sub = g.op("Mul", sub_v, sub_v)
    keepdim_mean = 0 if dim is None else keepdim
    var = g.op("ReduceMean", sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
    # Correct bias in calculating variance, by dividing it over (N - correction) instead on N
    if correction is None:
        correction = 1
    if correction != 0:
        num_elements = g.op(
            "Cast", num_elements, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"]
        )
        one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
        mul = g.op("Mul", var, num_elements)
        var = g.op("Div", mul, g.op("Sub", num_elements, one))
    return var, mean


def std(g, input, *args):
    var, _ = var_mean(g, input, *args)
    return g.op("Sqrt", var)


def var(g, input, *args):
    var, _ = var_mean(g, input, *args)
    return var


# var_mean (and all variance-related functions) has multiple signatures, so need to manually figure
# out the correct arguments:
# aten::var_mean(Tensor self, bool unbiased)
# aten::var_mean(Tensor self, int[1] dim, bool unbiased, bool keepdim=False)
# aten::var_mean(Tensor self, int[1]? dim=None, *, int? correction=None, bool keepdim=False)
def var_mean(g, input, *args):
    if len(args) == 1:
        return _var_mean(g, input, None, args[0], None)
    else:
        return _var_mean(g, input, *args)


def std_mean(g, input, *args):
    var, mean = var_mean(g, input, *args)
    return g.op("Sqrt", var), mean


@symbolic_helper.parse_args("v", "is", "i")
def logsumexp(g, input, dim, keepdim):
    return g.op("ReduceLogSumExp", input, axes_i=dim, keepdims_i=keepdim)


def arange(g, *args):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("arange", *args)

    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    def _float_step_convert(range_tensor):
        if symbolic_helper._is_fp(range_tensor):
            range_tensor = g.op(
                "Cast",
                g.op("Ceil", range_tensor),
                to_i=symbolic_helper.scalar_type_to_onnx[4],
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
            "Cast", arange_tensor, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
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
            "Cast", arange_tensor, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
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
            "Cast", arange_tensor, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
        )
    else:
        raise NotImplementedError(
            "Unknown aten::arange signature taking " + str(len(args)) + " arguments."
        )


def linspace(g, start, end, steps, dtype, layout, device, pin_memory):
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    step = div(
        g,
        sub(g, end, start),
        sub(g, steps, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))),
    )
    return add(g, mul(g, range_tensor, step), start)


def lift(g, self):
    # at::lift() is a no-op from the perspective of tracing for onnx
    return self


def masked_fill(g, self, mask, value):
    mask = _cast_Bool(g, mask, False)  # type: ignore[name-defined]
    value = symbolic_helper._maybe_get_scalar(value)
    return g.op("Where", mask, symbolic_helper._if_scalar_type_as(g, value, self), self)


def index(g, self, index):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index", self, index, overload_name="Tensor")

    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    def try_mask_to_index(index):
        if not symbolic_helper._is_none(index) and (
            index.type().scalarType() == "Byte" or index.type().scalarType() == "Bool"
        ):
            if GLOBALS.export_onnx_opset_version < 9:
                raise RuntimeError(
                    "Exporting masked indices are only supported after ONNX opset 9."
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
                raise NotImplementedError(
                    "Unsupported aten::index operator of advanced indexing on tensor of unknown rank, "
                    + "try turning on shape and type propagate during export: "
                    + "torch.onnx._export(..., propagate=True)."
                )
            # TODO: If indexing is supported natively in ONNX in future opsets,
            #       update the warning to recommend exporting with higher opset version.
            warnings.warn(
                "Exporting aten::index operator of advanced indexing in opset "
                + str(GLOBALS.export_onnx_opset_version)
                + " is achieved by combination of multiple ONNX operators, "
                + "including Reshape, Transpose, Concat, and Gather. "
                + "If indices include negative values, the exported graph will produce incorrect results."
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


@symbolic_helper.parse_args("v", "v", "is", "i", "v")
def linalg_norm(
    g,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: List[int],
    keepdim: int,
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
                "dim", "Input rank must be known at export time."
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


@symbolic_helper.parse_args("v", "f", "is", "i", "v")
def linalg_vector_norm(
    g,
    self: torch._C.Value,
    ord: float,
    dim: List[int],
    keepdim: int,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    if dim is None:
        self = symbolic_helper._reshape_helper(g, self, [-1])
        keepdim = 0

    if ord == math.inf:
        result = g.op("ReduceMax", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim)
    elif ord == -math.inf:
        result = g.op("ReduceMin", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim)
    elif ord == 0:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "linalg_vector_norm", 9, 11, "ord=0 not supported"
        )
    else:
        ord_op = g.op("Constant", value_t=torch.tensor(ord, dtype=torch.float32))
        result = symbolic_helper._reducesum_helper(
            g, g.op("Pow", g.op("Abs", self), ord_op), axes_i=dim, keepdims_i=keepdim
        )
        result = g.op(
            "Pow",
            result,
            g.op(
                "Div",
                g.op("Constant", value_t=torch.tensor(1, dtype=torch.float32)),
                ord_op,
            ),
        )
    return result


@symbolic_helper.parse_args("v", "v", "is", "i", "v")
def linalg_matrix_norm(
    g,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: List[int],
    keepdim: int,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html
    ord_value = symbolic_helper._parse_arg(ord, "s")
    if ord_value == "fro":
        return frobenius_norm(g, self, dim, keepdim)
    elif ord_value == "nuc":
        return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==nuc")
    else:
        ord_value = symbolic_helper._parse_arg(ord, "f")
        if ord_value is None:
            return frobenius_norm(g, self, dim, keepdim)
        if ord_value == 2 or ord_value == -2:
            # ord = 2/-2 unimplemented due to lack of operators
            # used to calculate singular values
            return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==2")
        # Wrap the dim vector to handle neagtive dim values
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "linalg.matrix_norm", "Input rank must be known at export time."
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


@symbolic_helper.parse_args("v", "v", "i")
def linalg_cross(g, input, other, dim=-1):
    return cross(g, input, other, dim)


@symbolic_helper.parse_args("v", "is", "i")
def frobenius_norm(g, self, dim=None, keepdim=False):
    sqr = g.op("Mul", self, self)
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, axes_i=dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)


@symbolic_helper.parse_args("v", "i", "b", "v")
def multinomial(g, input, num_samples, replacement=False, generator=None):
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Multinomial", "generator is not supported for multinomial"
        )
    if not replacement and num_samples > 1:
        symbolic_helper._unimplemented(
            "Multinomial",
            "replacement=False when num_samples > 1 is not supported for multinomial",
        )

    log_input = log(g, input)
    return g.op(
        "Multinomial",
        log_input,
        dtype_i=symbolic_helper.cast_pytorch_to_onnx["Long"],
        sample_size_i=num_samples,
    )


def baddbmm(g, self, batch1, batch2, beta, alpha):
    dtype = self.type().scalarType()
    batch_mul = matmul(g, batch1, batch2)
    mul_a = mul(
        g,
        batch_mul,
        g.op("Cast", alpha, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype]),
    )
    mul_b = mul(
        g, self, g.op("Cast", beta, to_i=symbolic_helper.cast_pytorch_to_onnx[dtype])
    )
    return add(g, mul_a, mul_b)


@symbolic_helper.parse_args("v", "s")
def meshgrid(g, tensor_list, indexing: Optional[str] = None):
    if indexing is None:
        indexing = "ij"
    elif indexing not in {"ij", "xy"}:
        raise ValueError(f"Unsupported indexing: {indexing}")
    if indexing == "xy":
        tensor_list[0], tensor_list[1] = tensor_list[1], tensor_list[0]
    tensors = [
        symbolic_helper._reshape_helper(
            g, t, g.op("Constant", value_t=torch.LongTensor([-1]))
        )
        for t in symbolic_helper._unpack_list(tensor_list)
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


def remainder(g, input, other):
    div = _floor_divide(g, input, other)
    quo = g.op("Mul", div, other)
    return g.op("Sub", input, quo)


@symbolic_helper.parse_args("v", "s")
def gelu(g, self: torch._C.Value, approximate: str = "none"):
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


@symbolic_helper.parse_args("v", "i", "v", "v", "f", "i")
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
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
        return symbolic_helper._unimplemented("group_norm", "unknown input rank")
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
        value_t=torch.tensor([1.0] * num_groups).type(
            "torch." + input.type().scalarType() + "Tensor"
        ),
    )
    bias_ = g.op(
        "Constant",
        value_t=torch.tensor([0.0] * num_groups).type(
            "torch." + input.type().scalarType() + "Tensor"
        ),
    )

    norm_reshaped = g.op(
        "InstanceNormalization", input_reshaped, weight_, bias_, epsilon_f=eps
    )
    norm = symbolic_helper._reshape_helper(g, norm_reshaped, g.op("Shape", input))

    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor([1.0]).type(
            "torch." + input.type().scalarType() + "Tensor"
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().mustBeNone():
        bias_value = torch.tensor([0.0]).type(
            "torch." + input.type().scalarType() + "Tensor"
        )
        bias = g.op("Constant", value_t=bias_value)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    axes = list(range(1, input_rank - 1))
    return add(
        g,
        mul(g, norm, symbolic_helper._unsqueeze_helper(g, weight, axes)),
        symbolic_helper._unsqueeze_helper(g, bias, axes),
    )


@symbolic_helper.parse_args("v", "v", "i")
def _weight_norm(g, weight_v, weight_g, dim):
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
    elif symbolic_helper.is_caffe2_aten_fallback():
        return g.at("_weight_norm", weight_v, weight_g, dim_i=dim)
    else:
        raise RuntimeError(
            "Unsupported: ONNX export of _weight_norm for tensor " "of unknown rank."
        )


def dim(g, self):
    """Implement the dim functionality available for a pytorch tensor in ONNX"""
    # ONNX does not support dim directly in this opset so we can use 2 ops to get the info
    shape = g.op("Shape", self)
    return g.op("Size", shape)


def __getitem_(g, self, i):
    return select(g, self, g.op("Constant", value_t=torch.tensor([0])), i)


def item(g, self):
    return self


def take(g, self, index):
    self_flattened = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    out = index_select(g, self_flattened, 0, index)
    out = reshape_as(g, out, index)
    return out


def _kl_div_log_target_impl(g, input, target):
    diff_ = sub(g, target, input)
    exp_ = exp(g, target)
    output = mul(g, exp_, diff_)
    return output


def _kl_div_non_log_target_impl(g, input, target):
    log_ = log(g, target)
    diff_ = sub(g, log_, input)
    output_pos = mul(g, target, diff_)
    zeros_ = zeros_like(g, output_pos)
    mask_ = gt(g, target, g.op("Constant", value_t=torch.tensor(0)))
    output = where(g, mask_, output_pos, zeros_)
    return output


@symbolic_helper.parse_args("v", "v", "i", "b")
def kl_div(g, input, target, reduction, log_target):
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
            "kl_div with reduction other than none, mean, or sum."
        )


@symbolic_helper.parse_args("v", "v", "is", "i")
def as_strided(g, self, sizes, strides, offset=None):
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


def __derive_index(g, index, start, step):
    return g.op("Add", start, g.op("Mul", index, step))


# Source code for aten op can be found here: pytorch/torch/csrc/jit/runtime/register_prim_ops.cpp
# if (step > 0 && lo < hi) {
#   push(stack, 1 + (hi - 1 - lo) / step);
# } else if (step < 0 && lo > hi) {
#   push(stack, 1 + (lo - 1 - hi) / (0 - step));
# } else {
#  push(stack, 0);
# }
def __range_length(g, lo, hi, step):
    sub = g.op("Sub", hi, lo)
    div = g.op("Ceil", true_divide(g, sub, step))
    return g.op("Cast", div, to_i=symbolic_helper.cast_pytorch_to_onnx["Long"])


def linear(g, input, weight, bias):
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


@symbolic_helper.parse_args("v", "b", "i", "v", "v", "v", "v")
def hann_window(
    g,
    window_length,
    periodic=True,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    requires_grad=False,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
        if not dtype or not dtype.is_floating_point:
            dtype = torch.float
        dtype = symbolic_helper.scalar_type_to_pytorch_type.index(dtype)

    n_array = arange(g, window_length, 4, None, None, None)
    output = g.op("Cast", n_array, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
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
        to_i=symbolic_helper.scalar_type_to_onnx[dtype],
    )

    return output


def mv(g, self, vec):
    return matmul(g, self, vec)


def dot(g, self, other):
    return matmul(g, self, other)


@symbolic_helper.parse_args("v", "t", "t")
def movedim(g, self, source, destination):
    # This is a pythonic implementation mostly taken from aten/src/ATen/native/TensorShape.cpp::movedim
    source = source.view(-1)
    destination = destination.view(-1)

    assert source.size() == destination.size()

    if (source == destination).all():
        return self

    self_rank = symbolic_helper._get_tensor_rank(self)

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


@symbolic_helper.parse_args("v", "v")
def fill(g, self, value):
    dtype = self.type().scalarType()
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    else:
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )

    return full_like(g, self, value, dtype)


def index_add(g, self, dim, index, other, alpha=None):
    warnings.warn(
        "Warning: ONNX export does not support duplicated values in 'index' field, "
        + "this will cause the ONNX model to be incorrect."
    )

    # ONNX does not support "alpha" argument, unlike aten index_add
    # See: https://github.com/pytorch/pytorch/pull/65993#issuecomment-953151102 for more context
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        return symbolic_helper._unimplemented("index_add", "alpha != 1")

    dim = symbolic_helper._maybe_get_const(dim, "i")
    if dim is None:
        raise NotImplementedError(
            "ONNX export does NOT support exporting 'index_add_()' function with "
            + "unknown 'dim' value."
        )

    self_dim_rank = symbolic_helper._get_tensor_rank(self)
    other_dim_rank = symbolic_helper._get_tensor_rank(other)

    if self_dim_rank is None or other_dim_rank is None:
        raise NotImplementedError(
            "ONNX export does NOT support exporting 'index_add_()' function while "
            + "the rank of self tensor or tensor to be added is unknown."
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
            raise NotImplementedError(
                "ONNX export does NOT support exporting 'index_add_()' function with "
                + "duplicated values in 'index' parameter yet."
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


@symbolic_helper.parse_args("v", "is", "is")
def roll(g, self, shifts, dims):
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


@symbolic_helper.parse_args("v", "v", "i")
def cross(g, input, other, dim=None):
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


def cdist(g, x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    # X1.shape = (B * P * D), X2.shape = (B * R * D)
    # In order to respect numpy style broadcasting as demonstrated in
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    # we unsqueeze both input tensors
    # Currently we ignore the 'compute_mode' variable as we use default to
    # using matrix multiplication to calculate the euclidean distance
    rank = symbolic_helper._get_tensor_rank(x1)
    broadcasted_x1 = symbolic_helper._unsqueeze_helper(g, x1, [rank - 1])
    broadcasted_x2 = symbolic_helper._unsqueeze_helper(g, x2, [rank - 2])
    return pairwise_distance(
        g, broadcasted_x1, broadcasted_x2, p, eps=1e-06, keepdim=False
    )


def lerp(g, self, end, weight):
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


def broadcast_tensors(g, self):
    all_tensors = symbolic_helper._unpack_list(self)
    t_with_final_shape = zeros_like(g, all_tensors[0])

    # Add operator supports multidirectional broadcasting. So we leverage this function
    # to infer the final shape generated by the broadcast.
    for t in all_tensors:
        t_with_final_shape = add(g, t_with_final_shape, t)

    t_list = [expand_as(g, t, t_with_final_shape) for t in all_tensors]
    return g.op("prim::ListConstruct", *t_list)


class Prim:
    domain = "prim"

    @staticmethod
    def ConstantSplit(g, self, split_size, dim):
        size = symbolic_helper._get_tensor_dim_size(self, dim)
        if size is None:
            return symbolic_helper._unimplemented(
                "prim::ConstantSplit", "unknown dimension size"
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
    @staticmethod
    def ConstantChunk(g, self, chunks, dim):
        dim_size = symbolic_helper._get_tensor_dim_size(self, dim)
        if dim_size is None:
            return symbolic_helper._unimplemented(
                "prim::ConstantChunk", "unknown dimension size"
            )
        split_size = (dim_size + chunks - 1) // chunks
        return Prim.ConstantSplit(g, self, split_size, dim)

    @staticmethod
    def shape(g, self):
        return g.op("Shape", self)

    @staticmethod
    def max(g, self, other):
        return op_with_optional_float_cast(g, "Max", self, other, opset_before=12)

    @staticmethod
    def min(g, self, other=None):
        if not other:
            if symbolic_helper._is_packed_list(self):
                self = stack(g, self, g.op("Constant", value_t=torch.tensor([0])))
            return min(g, self)
        return min(g, self, other)

    @staticmethod
    def data(g, self):
        return self

    @staticmethod
    def ListConstruct(g, *inputs, **kwargs):
        return None

    @staticmethod
    def ListUnpack(g, *inputs, **kwargs) -> Optional[List[_C.Value]]:
        if len(inputs) == 1 and inputs[0].node().kind() == "prim::ListConstruct":
            # Cancel the previous node if it is ListConstruct by returning its inputs
            # TODO(justinchuby): Use a public method in the helper module
            return symbolic_helper._unpack_list(inputs[0])

        return None

    @staticmethod
    def TupleConstruct(g, *inputs, **kwargs):
        return None

    @staticmethod
    def Uninitialized(g, *inputs, **kwargs):
        return None

    # exists to refine the type of the Value
    # if x is an optional Tensor, unchecked_cast will cast
    # x to Tensor, so the rest of the graph knows that x is a Tensor
    # this doesn't do anything in runtime and is a noop in ONNX
    @staticmethod
    def unchecked_cast(g, self):
        return self

    @staticmethod
    def dtype(g, self):
        dtype = symbolic_helper._try_get_scalar_type(self)
        if dtype is None:
            dtype = "Float"
        dtype = symbolic_helper.scalar_type_to_onnx.index(
            symbolic_helper.cast_pytorch_to_onnx[dtype]
        )
        return g.op("Constant", value_t=torch.tensor(dtype))

    # tolist is currently supported only for 1D input tensors.
    # dim_val and elem_ty_val represent dimension and type annotations
    # that need to match dimension and type of the input tensor.
    @staticmethod
    def tolist(g, input, dim_val, elem_ty_val):
        dim = symbolic_helper._maybe_get_const(dim_val, "i")
        if dim > 1:
            return symbolic_helper._unimplemented("prim::tolist", "dim_val > 1")
        return input

    # -----------------------------------------------------------------------------
    # Symbolic functions that need extra context
    # -----------------------------------------------------------------------------
    @staticmethod
    def device(ctx: SymbolicContext, g: _C.Graph, *inputs, **kwargs) -> None:
        output_type = ctx.cur_node.output().type()
        if isinstance(output_type, _C.DeviceObjType):
            return None

        return symbolic_helper._unimplemented(
            "prim::device",
            f"output type should be 'DeviceObjType', not '{output_type.kind()}'",
        )

    @staticmethod
    def Loop(ctx: SymbolicContext, g, *inputs, **attrs):
        n = ctx.cur_node
        env = ctx.env
        params_dict = ctx.params_dict

        operator_export_type = GLOBALS.operator_export_type
        opset_version = GLOBALS.export_onnx_opset_version

        new_op_outputs = g.op("Loop", *inputs, outputs=n.outputsSize())
        new_node = (
            new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
        )
        for b in n.blocks():
            new_block = new_node.addBlock()
            # Copy input metadata to subblock
            #
            #   prim::Loop(iter, cond, input_1, ..., input_n)
            #     block0(iter, input_1, ..., input_n)
            #
            # For `Loop` node, copy metadata for `iter`, `input_1`, ..., `input_n`.
            for i, b_in in enumerate(b.inputs()):
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
                b, new_block, operator_export_type, env, False  # type:ignore[arg-type]
            )
        new_op_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
            new_node, opset_version
        )
        # Run shape type inference for Loop after subblock is converted.
        if GLOBALS.onnx_shape_inference:
            torch._C._jit_pass_onnx_node_shape_type_inference(
                new_node, params_dict, opset_version
            )
        return new_op_outputs

    @staticmethod
    def If(ctx: SymbolicContext, g, *inputs, **attrs):
        n = ctx.cur_node
        block = ctx.onnx_block
        env = ctx.env
        params_dict = ctx.params_dict

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
            input_flag = inputs[0].node()["value"].tolist()
            const_value = (
                all(input_flag) if isinstance(input_flag, list) else bool(input_flag)
            )
            block_idx = 0 if const_value else 1
            current_b = list(n.blocks())[block_idx]
            env = torch._C._jit_pass_onnx_block(
                current_b,
                block,
                operator_export_type,  # type:ignore[arg-type]
                env,  # type:ignore[arg-type]
                True,
            )
            if_output_list = list(n.outputs())
            current_b_list = list(current_b.outputs())

            final_b_list = []
            for idx in range(len(if_output_list)):
                if current_b_list[idx] not in env:
                    raise RuntimeError(
                        f"The sub block ATen output {current_b_list[idx]} is not in env."
                    )  # type:ignore[operator]
                onnx_b = env[current_b_list[idx]]
                final_b_list.append(onnx_b)
            return final_b_list
        else:
            new_op_outputs = g.op("If", *inputs, outputs=n.outputsSize())
            new_node = (
                new_op_outputs[0].node()
                if n.outputsSize() > 1
                else new_op_outputs.node()
            )
            for b in n.blocks():
                new_block = new_node.addBlock()
                torch._C._jit_pass_onnx_block(
                    b,
                    new_block,
                    operator_export_type,  # type:ignore[arg-type]
                    env,
                    False,
                )
            new_op_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
                new_node, opset_version
            )
            # Run shape type inference for If after subblock is converted.
            if GLOBALS.onnx_shape_inference:
                torch._C._jit_pass_onnx_node_shape_type_inference(
                    new_node, params_dict, opset_version
                )
            return new_op_outputs

    @staticmethod
    def Constant(ctx: SymbolicContext, g, *inputs, **attrs):
        n = ctx.cur_node

        if n.mustBeNone():
            return None
        # This must go before checking for string values, because some device constants
        # have string values, but we want to keep them as unconverted Device types so
        # that eq() can work on them.
        if isinstance(n.output().type(), _C.DeviceObjType):
            return None
        if n.kindOf("value") == "t":
            return g.op("Constant", value_t=n["value"])
        if n.kindOf("value") == "s":
            return g.op("Constant", value_s=n["value"])
        elif n.output().type().isSubtypeOf(
            _C.ListType.ofInts()
        ) or n.output().type().isSubtypeOf(_C.ListType.ofFloats()):
            return g.op("Constant", value_t=torch.tensor(n["value"]))
        else:
            raise RuntimeError(
                f"Unsupported prim::Constant kind: `{n.kindOf('value')}`. Send a bug report."
            )


class Onnx:
    domain = "onnx"

    # -----------------------------------------------------------------------------
    # Symbolic functions that need extra context
    # -----------------------------------------------------------------------------
    @staticmethod
    def Placeholder(ctx: SymbolicContext, g, *inputs, **attrs):
        n = ctx.cur_node
        block = ctx.onnx_block
        env = ctx.env

        return torch._C._jit_onnx_convert_pattern_from_subblock(block, n, env)
