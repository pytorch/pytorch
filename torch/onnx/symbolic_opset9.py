import torch
from torch._C import ListType, OptionalType
from torch.nn.modules.utils import _single, _pair, _triple

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from functools import partial
from functools import wraps

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _unimplemented

from typing import Optional

import numpy
import math
import warnings


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 9
# Opset 9 is supported by ONNX release 1.4.1
# release on 01/23/19


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

# used to represent "missing" optional inputs
def unused(g):
    n = g.op("prim::Constant")
    n.setType(OptionalType.ofTensor())
    return n

def _shape_as_tensor(g, input):
    return g.op('Shape', input)


def _reshape_from_tensor(g, input, shape):
    return g.op('Reshape', input, shape)


def reshape(g, self, shape):
    return view(g, self, shape)


def reshape_as(g, self, other):
    shape = g.op('Shape', other)
    return reshape(g, self, shape)


def add(g, self, other, alpha=None):
    if sym_help._is_value(self) and sym_help._is_tensor_list(self):
        return sym_help._onnx_opset_unsupported_detailed('Add', 9, 11, 'Add between list of tensors not supported')

    # default alpha arg is to allow no-alpha add (aten add st overload no alpha)
    if alpha and sym_help._scalar(sym_help._maybe_get_scalar(alpha)) != 1:
        return _unimplemented("add", "alpha != 1")
    return g.op("Add", self, other)


def sub(g, self, other, alpha=None):
    # default alpha arg is to allow no-alpha sub (aten sub st overload no alpha)
    if alpha and sym_help._scalar(sym_help._maybe_get_scalar(alpha)) != 1:
        return _unimplemented("sub", "alpha != 1")
    return g.op("Sub", self, other)


def rsub(g, self, other, alpha=None):
    return sub(g, other, self, alpha=alpha)


def mul(g, self, other):
    return g.op("Mul", self, other)


def div(g, self, other):
    return true_divide(g, self, other)


def floor_divide(g, self, other):
    out = g.op('Div', self, other)
    # the correct operation is truncate, which is not supported in ONNX,
    # we cannot call floor since it will behave differently for negative numbers
    # (eg. -0.1 should become -0 )
    # - if scalar_type information are not available, assume that
    # we need to call floor (treat as float)
    out = g.op("Cast", out, to_i=sym_help.cast_pytorch_to_onnx['Long'])

    # Matching PyTorch's behavior:
    # - if self is fp the output's type is self's type
    # - if self is not fp and other is fp, the output is of type 'Float'
    # - self is not fp and other is not fp, the output's type is self's output type
    # - the output type defaults to Float
    scalar_type = self.type().scalarType()

    if scalar_type is not None:
        if not sym_help._is_fp(self) and \
           other.type().scalarType() is not None and \
           sym_help._is_fp(other):
            out = g.op("Cast", out, to_i=sym_help.cast_pytorch_to_onnx['Float'])
        else:
            out = g.op("Cast", out, to_i=sym_help.cast_pytorch_to_onnx[scalar_type])
    else:
        out = g.op("Cast", out, to_i=sym_help.cast_pytorch_to_onnx['Float'])
    return out


def floordiv(g, self, other):
    return floor_divide(g, self, other)

# Division where both inputs are cast to floating types
# If both inputs are floating, performs div as usual
# If only one input is a floating type, the other input is cast to its type
# If neither input is a floating type, both inputs are cast to the default scalar type
def true_divide(g, self, other):
    # Case 1: both values are floating
    # Performs div as usual
    if sym_help._is_fp(self) and sym_help._is_fp(other):
        return g.op("Div", self, other)

    # Case 2: self is floating, other is not
    # Casts other to self's dtype
    if sym_help._is_fp(self):
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
        return g.op("Div", self, other)

    # Case 3: other is floating, self is not
    # Casts self to other's dtype
    if sym_help._is_fp(other):
        self = g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[other.type().scalarType()])
        return g.op("Div", self, other)

    # Case 4: neither is floating
    # Casts both inputs to the default scalar type
    scalar_type = torch.get_default_dtype()
    onnx_scalar_type = sym_help.cast_pytorch_to_onnx['Float']
    assert scalar_type is torch.float or scalar_type is torch.double
    if torch.get_default_dtype() is torch.double:
        onnx_scalar_type = sym_help.cast_pytorch_to_onnx['Double']

    self = g.op("Cast", self, to_i=onnx_scalar_type)
    other = g.op("Cast", other, to_i=onnx_scalar_type)
    return g.op("Div", self, other)


def reciprocal(g, self):
    return g.op("Div", torch.ones(1), self)


@parse_args('v', 'i')
def cat(g, tensor_list, dim):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@parse_args('v', 'i')
def stack(g, tensor_list, dim):
    unsqueezed = [sym_help._unsqueeze_helper(g, t, [dim]) for t in sym_help._unpack_list(tensor_list)]
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


@parse_args('v', 'v', 'v', 't', 't')
def addmm(g, self, mat1, mat2, beta, alpha):
    dtype = None
    self_dtype = sym_help._try_get_scalar_type(self)
    mat1_dtype = sym_help._try_get_scalar_type(mat1)
    mat2_dtype = sym_help._try_get_scalar_type(mat2)
    if self_dtype is not None:
        dtype = self_dtype
    elif mat1_dtype is not None:
        dtype = mat1_dtype
    elif mat2_dtype is not None:
        dtype = mat2_dtype

    mat1_rank = sym_help._get_tensor_rank(mat1)
    mat2_rank = sym_help._get_tensor_rank(mat2)

    def isNotNoneAnd(v, u):
        return v is not None and v != u

    if dtype is not None and (isNotNoneAnd(mat1_rank, 2) or isNotNoneAnd(mat2_rank, 2)):
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
        dtype = sym_help.scalar_type_to_pytorch_type[dtype]

        res1 = g.op("MatMul", mat1, mat2)
        res2 = self

        alpha = sym_help._scalar(alpha)
        beta = sym_help._scalar(beta)

        if alpha != 1:
            alpha = g.op("Constant",
                         value_t=torch.tensor(alpha, dtype=dtype))
            res1 = g.op("Mul", res1, alpha)
        if beta != 1:
            beta = g.op("Constant",
                        value_t=torch.tensor(sym_help._scalar(beta), dtype=dtype))
            res2 = g.op("Mul", res2, beta)

        return g.op("Add", res1, res2)

    return g.op("Gemm", mat1, mat2, self, beta_f=sym_help._scalar(beta), alpha_f=sym_help._scalar(alpha))


def neg(g, self):
    return g.op("Neg", self)


def sqrt(g, self):
    return g.op("Sqrt", self)


def rsqrt(g, self):
    return g.op("Div", sym_help._if_scalar_type_as(g, torch.ones(1), self), sqrt(g, self))


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
        if not sym_help._is_fp(self) and not (dtype == 'Long'):
            self = _cast_Long(g, self, False)  # type: ignore
    return self


def _reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return g.op(onnx_op_name, self, keepdims_i=0)
        else:
            # dim-reduce path
            desc = 'is' if allow_multi_dim_support else 'i'
            dim, keepdim = sym_help._get_const(dim, desc, 'dim'), sym_help._get_const(keepdim, 'i', 'keepdim')
            dim_list = dim if allow_multi_dim_support else [dim]
            return g.op(onnx_op_name, self, axes_i=dim_list, keepdims_i=keepdim)
    return symbolic



def overload_by_arg_count(fn):
    @wraps(fn)
    def wrapper(g, *args):
        overloads = fn(g, *args)
        last_exception = None
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        raise NotImplementedError("Unknown aten::{} signature".format(fn.__name__))
    return wrapper


def _reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True):
    symbolic = _reduce_op_symbolic(onnx_op, allow_multi_dim_support=allow_multi_dim_support)

    @overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @parse_args('v', 'none')
        def reduce_nodim(g, self, dtype):
            if dtype.node().kind() != 'prim::Constant':
                return _unimplemented(name, "dtype")
            return symbolic(g, self)

        dim_desc = 'is' if allow_multi_dim_support else 'i'

        @parse_args('v', dim_desc, 'i', 'none')
        def reduce_dim(g, self, dim, keepdim, dtype):
            if dtype.node().kind() != 'prim::Constant':
                return _unimplemented(name, "dtype")
            return symbolic(g, self, dim, keepdim)
        return reduce_nodim, reduce_dim
    return reduce


sum = _reduce_with_dtype('ReduceSum', 'sum')
mean = _reduce_with_dtype('ReduceMean', 'mean')
prod = _reduce_with_dtype('ReduceProd', 'prod', allow_multi_dim_support=False)  # torch.prod does not support multidimensional 'dim'


@parse_args('v', 'i', 'none')
def cumsum(g, input, dim, dtype):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        if dtype.node().kind() != 'prim::Constant':
            return _unimplemented(name, "dtype")
        return g.op("ATen", input, operator_s="cumsum", dim_i=dim)
    else:
        sym_help._onnx_opset_unsupported('cumsum', 9, 11)


def _sample_dirichlet(g, self, generator):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        if not sym_help._is_none(generator):
            return _unimplemented('_sample_dirichlet',
                                  'We are not able to export generator')
        return g.op("ATen", self, operator_s="_sample_dirichlet")
    else:
        return sym_help._onnx_unsupported('_sample_dirichlet')


def _standard_gamma(g, self, generator):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        if not sym_help._is_none(generator):
            return _unimplemented('_standard_gamma',
                                  'We are not able to export generator')
        return g.op("ATen", self, operator_s="_standard_gamma")
    else:
        return sym_help._onnx_unsupported('_standard_gamma')


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def expand(g, self, size, implicit):
    size = sym_help._maybe_get_const(size, 'is')
    if not sym_help._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif sym_help._is_packed_list(size):
        # Expand with -1 dim value means dim is unchanged.
        # Since onnx::expand supports two-way broadcasting,
        # -1 dim value can be exported to onnx as 1
        size = view(g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1])))
    dtype = 4  # dim type is int64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


def expand_as(g, self, other):
    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op("Gather", weight, indices)


@parse_args('v', 'v', 'v', 'i', 'i', 'i', 'v', 'i')
def embedding_bag(g,
                  embedding_matrix,
                  indices,
                  offsets,
                  scale_grad_by_freq,
                  mode,
                  sparse,
                  per_sample_weights,
                  include_last_offset):
    if not sym_help._is_none(per_sample_weights):
        return sym_help._onnx_unsupported('embedding_bag  with per_sample_weights')
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen",
                    embedding_matrix,
                    indices,
                    offsets,
                    operator_s="embedding_bag",
                    outputs=4,
                    scale_grad_by_freq_i=scale_grad_by_freq,
                    mode_i=mode,
                    sparse_i=sparse,
                    include_last_offset_i=include_last_offset)
    else:
        return sym_help._onnx_unsupported('embedding_bag')


def size(g, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    if sym_help._maybe_get_const(dim, 'i') < 0:
        rank = sym_help._get_tensor_rank(self)
        if rank is not None:
            dim = sym_help._maybe_get_const(dim, 'i') + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    return sym_help._size_helper(g, self, dim)


@parse_args('v', 'i', 'i')
def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    rank = sym_help._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return g.op("Transpose", self, perm_i=axes)
    else:
        # if we don't have dim information we cannot
        # output a permute so use ATen instead
        if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
            return g.op("ATen", self, operator_s="transpose", dim0_i=dim0, dim1_i=dim1)
        else:
            raise RuntimeError('Unsupported: ONNX export of transpose for tensor '
                               'of unknown rank.')


@parse_args('v', 'is')
def permute(g, self, dims):
    if dims == list(range(0, len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


def view(g, self, size):
    size = sym_help._maybe_get_const(size, 'is')
    if sym_help._is_value(size):
        shape = size
    else:
        shape = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Reshape", self, shape)


def view_as(g, self, other):
    shape = g.op("Shape", other)
    return g.op("Reshape", self, shape)


def prim_ConstantSplit(g, self, split_size, dim):
    size = sym_help._get_tensor_dim_size(self, dim)
    if size is None:
        return _unimplemented('prim::ConstantSplit', 'unknown dimension size')
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=len(splits))


# TODO: It would be better to export this as a chunk directly, as this is
# less sensitive to changes in input size.
# TODO: Once we have proper scoping, stop reimplementing chunk, delete this
# method, and use the desugared version
def prim_ConstantChunk(g, self, chunks, dim):
    dim_size = sym_help._get_tensor_dim_size(self, dim)
    if dim_size is None:
        return _unimplemented('prim::ConstantChunk', 'unknown dimension size')
    split_size = (dim_size + chunks - 1) // chunks
    return prim_ConstantSplit(g, self, split_size, dim)


@parse_args('v', 'i', 'i', 'i')
def unsafe_chunk(g, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return sym_help._onnx_opset_unsupported_detailed('unsafe_chunk', 9, 11, 'Dynamic number of outputs not supported')
    size = sym_help._get_tensor_dim_size(self, dim)
    if size is None:
        return _unimplemented('unsafe_chunk', 'unknown dimension size')
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@parse_args('v', 'v', 'v', 'i')
def split(g, self, split_size_or_sizes, dim, _outputs=None):
    if not sym_help._is_split_static(split_size_or_sizes, _outputs):
        return sym_help._onnx_opset_unsupported_detailed('split', 9, 11, 'Dynamic number of outputs not supported')
    split_val = split_size_or_sizes.node()['value']
    if split_val.dim() > 0:
        return split_with_sizes(g, self, split_size_or_sizes, dim, _outputs)
    split_size = sym_help._get_const(split_size_or_sizes, 'i', 'split_size')
    dim = sym_help._get_const(dim, 'i', 'dim')

    size = sym_help._get_tensor_dim_size(self, dim)
    if size is None:
        return sym_help._onnx_opset_unsupported_detailed('split', 9, 11, 'Unknown dimension size not supported')
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


def unsafe_split(g, self, split_size_or_sizes, dim, _outputs=None):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@parse_args('v', 'is', 'i', 'i')
def split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    if not sym_help._is_split_static(split_sizes, _outputs):
        return sym_help._onnx_opset_unsupported_detailed('split_with_sizes', 9, 11, 'Dynamic number of outputs not supported')
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=_outputs)


def unsafe_split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@parse_args('v', 'i', 'i')
def unbind(g, self, dim=0, _outputs=None):
    if _outputs is None:
        return sym_help._onnx_opset_unsupported_detailed('unbind', 9, 11, 'Dynamic number of outputs not supported')

    outputs = g.op("Split", self, split_i=[1] * _outputs, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [sym_help._squeeze_helper(g, out, [dim]) for out in outputs]
    return squeezed_outputs


@parse_args('v', 'i', 'v')
def select(g, self, dim, index):
    index = sym_help._maybe_get_scalar(index)
    if (not sym_help._is_value(index)) and (index < 0):
        if index == -1:
            end_index = 9223372036854775807
        else:
            end_index = index + 1
        slice_node = sym_help._slice_helper(g, self, axes=[dim], starts=[index], ends=[end_index])
        return sym_help._squeeze_helper(g, slice_node, [dim])
    else:
        return g.op("Gather", self, index, axis_i=dim)


def square(g, self):
    return g.op("Mul", self, self)


def squeeze(g, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    squeeze_dim = sym_help._get_const(dim, 'i', 'dim')
    # Handle negative dims
    if squeeze_dim < 0:
        rank = sym_help._get_tensor_rank(self)
        if rank is not None:
            warnings.warn("ONNX export squeeze with negative axis " + str(squeeze_dim) +
                          " might cause the onnx model to be incorrect. " +
                          "Negative axis is not supported in ONNX. " +
                          "Axis is converted to " + str(squeeze_dim + rank) +
                          " based on input shape at export time. " +
                          "Passing an tensor of different rank in execution will be incorrect.")
            squeeze_dim += rank
        else:
            return _unimplemented('squeeze', 'negative axis with unknown input rank')

    dim_size = sym_help._get_tensor_dim_size(self, squeeze_dim)
    if dim_size is None:
        warnings.warn("This model contains a squeeze operation on dimension " + str(squeeze_dim) + " on an input " +
                      "with unknown shape. Note that if the size of dimension " + str(squeeze_dim) + " of the input " +
                      "is not 1, the ONNX model will return an error. Opset version 11 supports squeezing on " +
                      "non-singleton dimensions, it is recommended to export this model using opset " +
                      "version 11 or higher.")
        return sym_help._squeeze_helper(g, self, axes_i=[squeeze_dim])
    if dim_size > 1:
        warnings.warn("This model contains a squeeze operation on dimension " + str(squeeze_dim) + ". The size of " +
                      "this dimension in the given input is " + str(dim_size) + ". The model will " +
                      "be exported without the squeeze node. If the model is intended to be used with dynamic " +
                      "input shapes, please use opset version 11 to " +
                      "export the model.")
        return self

    warnings.warn("This model contains a squeeze operation on dimension " + str(squeeze_dim) + ". If the model is " +
                  "intended to be used with dynamic input shapes, please use opset version 11 to export the model.")
    return sym_help._squeeze_helper(g, self, axes_i=[squeeze_dim])

def prelu(g, self, weight):
    self_rank = sym_help._get_tensor_rank(self)
    if self_rank is not None and self_rank > 2:
        weight = sym_help._unsqueeze_helper(g, weight, list(range(1, self_rank - 1)))
    return g.op("PRelu", self, weight)


def silu(g, input):
    return g.op('Mul', input, g.op('Sigmoid', input))


def relu(g, input):
    return g.op("Relu", input)


def ceil(g, input):
    return g.op("Ceil", input)


def floor(g, input):
    return g.op("Floor", input)


def _len(g, self):
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return sym_help._squeeze_helper(g, sz_0, [0])


@parse_args('v', 't', 't')
def threshold(g, self, threshold, value):
    # See Note [Export inplace]
    if sym_help._scalar(threshold) != 0:
        return _unimplemented("threshold", "non-zero threshold")
    if sym_help._scalar(value) != 0:
        return _unimplemented("threshold", "non-zero value")
    return g.op("Relu", self)


def leaky_relu(g, input, negative_slope, inplace=False):
    negative_slope = sym_help._get_const(negative_slope, 't', 'negative_slope')
    # See Note [Export inplace]
    # TODO: Talk to ONNX about unconditional cast of scalar to float
    return g.op("LeakyRelu", input, alpha_f=sym_help._scalar(negative_slope))


@parse_args('v', 'i')
def glu(g, input, dim):
    dim_size = sym_help._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op('Split', input, axis_i=dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))


@parse_args('v', 'i', 'none')
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
    input_dim = sym_help._get_tensor_rank(input)
    if input_dim is not None:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        if dim < 0:
            dim = input_dim + dim

        is_transpose_required = (input_dim != dim + 1)

        if is_transpose_required:
            axes = list(range(input_dim))
            axes[dim], axes[-1] = axes[-1], axes[dim]
            input = g.op("Transpose", input, perm_i=axes)
            dim = input_dim - 1

        softmax = g.op('Softmax', input, axis_i=dim)
        if dtype and dtype.node().kind() != 'prim::Constant':
            parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
            softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])

        if is_transpose_required:
            softmax = g.op("Transpose", softmax, perm_i=axes)
        return softmax

    # Apply max normalization.
    input = g.op('Sub', input, g.op('ReduceMax', input, axes_i=[dim], keepdims_i=1))

    exp = g.op('Exp', input)
    sum = sym_help._reducesum_helper(g, exp, axes_i=[dim])
    softmax = g.op('Div', exp, sum)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return softmax

@parse_args('v', 't', 'v')
def softplus(g, self, beta, threshold):
    if beta != 1:
        return _unimplemented("beta", "has to be 1")
    return g.op('Softplus', self)


def get_pool_ceil_padding(input, kernel_size, stride, padding):
    sizes = sym_help._get_tensor_sizes(input)
    dim = sizes[-len(padding):] if sizes is not None else None
    if dim is None or any([i is None for i in dim]):
        return _unimplemented(name, "input size not accessible")
    ceiled_output_dim = [int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i]))) + 1
                         for i in range(0, len(padding))]
    # ensure last pooling starts inside
    ceiled_output_dim = [ceiled_output_dim[i] - 1
                         if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
                         else ceiled_output_dim[i]
                         for i in range(0, len(ceiled_output_dim))]
    padding_ceil = [0
                    if (stride[i] == 1)
                    else
                    (kernel_size[i] - (dim[i] + 2 * padding[i] - ((ceiled_output_dim[i] - 1) * stride[i] + 1)))
                    for i in range(0, len(padding))]
    # ensure padding is not > kernel_size
    padding_ceil = [(int(padding_ceil[i]) if padding_ceil[i] < kernel_size[i] - 1 else int(kernel_size[i] - 1))
                    if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
                    else
                    int(padding_ceil[i])
                    for i in range(0, len(padding_ceil))]
    return padding_ceil


def _max_pool(name, tuple_fn, ndims, return_indices):
    @parse_args('v', 'is', 'is', 'is', 'is', 'i')
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if set(tuple_fn(dilation)) != {1}:
            return _unimplemented(name, "dilation")
        if not stride:
            stride = kernel_size
        padding = tuple(tuple_fn(padding))
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            padding = padding + tuple(numpy.add(padding_ceil, padding))
        else:
            padding = padding * 2
        kwargs = {
            'kernel_shape_i': tuple_fn(kernel_size),
            'pads_i': padding,
            'strides_i': tuple_fn(stride),
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
            _, flattened_indices = g.op("MaxPool", input, outputs=2,
                                        kernel_shape_i=[1 for _ in range(ndims)],
                                        strides_i=[1 for _ in range(ndims)])
            # convert indices to have non-flattened indices values
            s = sym_help._slice_helper(g, flattened_indices, axes=[2 + i for i in range(ndims)],
                                       starts=tuple_fn(0), ends=tuple_fn(1))
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d = _max_pool("max_pool1d", _single, 1, return_indices=False)
max_pool2d = _max_pool("max_pool2d", _pair, 2, return_indices=False)
max_pool3d = _max_pool("max_pool3d", _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool("max_pool1d_with_indices", _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool("max_pool2d_with_indices", _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool("max_pool3d_with_indices", _triple, 3, return_indices=True)


def _avg_pool(name, tuple_fn):
    @parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
        if not stride:
            stride = kernel_size
        padding = sym_help._avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name)
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
        if count_include_pad:
            input = g.op("Pad", input,
                         pads_i=((0,) * 2 + padding) * 2,
                         mode_s='constant',
                         value_f=0.)
            padding = (0,) * len(padding)
        if ceil_mode:
            padding = padding + tuple(numpy.add(padding_ceil, padding))
        else:
            padding = padding * 2
        output = g.op("AveragePool", input,
                      kernel_shape_i=tuple_fn(kernel_size),
                      strides_i=tuple_fn(stride),
                      pads_i=padding)
        return output
    return symbolic_fn


avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)


def _adaptive_pool(name, type, tuple_fn, fn=None):
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
            output_size = _parse_arg(output_size, 'is')
        except Exception:
            return sym_help._onnx_unsupported('adaptive pooling, since output_size is not constant.')
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        sizes = sym_help._get_tensor_sizes(input)
        try:
            dim = sizes[2:]
        except Exception:
            dim = None
        if dim is None or any([i is None for i in dim]):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return _unimplemented(name, 'input size not accessible')
        # verify if output size % input size = 0 for all dim
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
                return _unimplemented(name, 'output size that are not factor of input size')
            else:
                return sym_help._onnx_unsupported(name + ', since output size is not factor of input size')
        k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
        # call max_poolxd_with_indices to get indices in the output
        if type == "MaxPool":
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        output = g.op(type, input,
                      kernel_shape_i=tuple_fn(k),
                      strides_i=tuple_fn(k))
        return output
    return symbolic_fn


adaptive_avg_pool1d = _adaptive_pool('adaptive_avg_pool1d', "AveragePool", _single)
adaptive_avg_pool2d = _adaptive_pool('adaptive_avg_pool2d', "AveragePool", _pair)
adaptive_avg_pool3d = _adaptive_pool('adaptive_avg_pool3d', "AveragePool", _triple)

adaptive_max_pool1d = _adaptive_pool('adaptive_max_pool1d', "MaxPool", _single, max_pool1d_with_indices)
adaptive_max_pool2d = _adaptive_pool('adaptive_max_pool2d', "MaxPool", _pair, max_pool2d_with_indices)
adaptive_max_pool3d = _adaptive_pool('adaptive_max_pool3d', "MaxPool", _triple, max_pool3d_with_indices)


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
    padding = sym_help._maybe_get_const(padding, 'is')
    if sym_help._is_value(padding) and sym_help._is_packed_list(padding):
        input_list = sym_help._unpack_list(padding)
        try:
            padding = [sym_help._get_const(v, 'i', 'padding') for v in input_list]
        except Exception:
            return sym_help._onnx_opset_unsupported_detailed('Pad', 9, 11, 'The sizes of the padding must be constant')
    return padding

def constant_pad_nd(g, input, padding, value):
    mode = "constant"
    try:
        value = sym_help._get_const(value, 'f', 'value')
    except Exception:
        return sym_help._onnx_opset_unsupported_detailed('Pad', 9, 11, 'The value for the padding must be constant')

    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode, value_f=value)


def reflection_pad(g, input, padding):
    mode = "reflect"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


def replication_pad(g, input, padding):
    mode = "edge"
    padding = _convert_padding_node(padding)
    paddings = _prepare_onnx_paddings(sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = sym_help._get_interpolate_attributes(g, interpolate_mode, args)
        sym_help._interpolate_warning(interpolate_mode)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        if align_corners:
            return _unimplemented(name, "align_corners == True")
        if scales is None:
            scales = sym_help._interpolate_size_to_scales(g, input, output_size, dim)
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)
    return symbolic_fn


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")


def __interpolate(g, input, size, scale_factor, mode , align_corners, recompute_scale_factor):
    scales, mode = sym_help._interpolate_get_scales_and_mode(g, input, size, scale_factor,
                                                             mode , align_corners)
    return g.op("Upsample", input, scales, mode_s=mode)

@parse_args('v')
def bitwise_not(g, inp):
    if inp.type().scalarType() != 'Bool':
        return _unimplemented("bitwise_not", "non-bool tensor")
    return g.op("Not", inp)


def wrap_logical_op_with_cast_to(to_type):
    def decorator(fn):
        def wrap_with_cast(g, input, other):
            return g.op("Cast", fn(g, input, other), to_i=sym_help.cast_pytorch_to_onnx[to_type])
        return wrap_with_cast
    return decorator


def wrap_logical_op_with_cast_to_and_from(to_type):
    def decorator(fn):
        def wrap_with_cast(g, input, other):
            to_cast_func = globals()['_cast_{}'.format(to_type)]
            from_cast_func = wrap_logical_op_with_cast_to(input.type().scalarType())(fn)
            return from_cast_func(g, to_cast_func(g, input, False), to_cast_func(g, other, False))
        return wrap_with_cast
    return decorator


def wrap_logical_op_with_negation(func):
    def wrap_with_not(g, input, other):
        return g.op("Not", func(g, input, other))
    return wrap_with_not


def eq(g, self, other):
    return g.op("Equal", self, other)


@wrap_logical_op_with_negation
def ne(g, self, other):
    return g.op("Equal", self, other)


def gt(g, input, other):
    return gt_impl(g, input, other)


def gt_impl(g, input, other):
    if input.type().scalarType() is not None and input.type().scalarType() == 'Bool' and \
            other.type().scalarType() is not None and other.type().scalarType() == 'Bool':
        input = g.op("Cast", input, to_i=sym_help.cast_pytorch_to_onnx['Int'])
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Int'])
    return g.op("Greater", input, other)


def lt(g, input, other):
    return lt_impl(g, input, other)


def lt_impl(g, input, other):
    if input.type().scalarType() is not None and input.type().scalarType() == 'Bool' and \
            other.type().scalarType() is not None and other.type().scalarType() == 'Bool':
        input = g.op("Cast", input, to_i=sym_help.cast_pytorch_to_onnx['Int'])
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Int'])
    return g.op("Less", input, other)


@wrap_logical_op_with_negation
def ge(g, input, other):
    return lt_impl(g, input, other)


@wrap_logical_op_with_negation
def le(g, input, other):
    return gt_impl(g, input, other)


@wrap_logical_op_with_cast_to_and_from('Bool')
def __and_(g, input, other):
    return g.op('And', input, other)


@wrap_logical_op_with_cast_to_and_from('Bool')
def __or_(g, input, other):
    return g.op('Or', input, other)


@wrap_logical_op_with_cast_to_and_from('Bool')
def logical_and(g, input, other):
    return g.op('And', input, other)


@wrap_logical_op_with_cast_to_and_from('Bool')
def logical_or(g, input, other):
    return g.op('Or', input, other)


@wrap_logical_op_with_cast_to_and_from('Bool')
def logical_xor(g, input, other):
    return g.op('Xor', input, other)


def __rshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])

    two = g.op('Constant', value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not sym_help._is_fp(self):
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Float'])
    two_pow = g.op('Pow', two, other)
    two_pow = g.op('Cast', two_pow, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
    rshift = g.op('Div', self, two_pow)
    return rshift


def __lshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])

    two = g.op('Constant', value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not sym_help._is_fp(self):
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Float'])
    two_pow = g.op('Pow', two, other)
    two_pow = g.op('Cast', two_pow, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
    lshift = g.op('Mul', self, two_pow)
    return lshift


@parse_args('v', 'v', 'v', 'i')
def where(g, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if condition.type().scalarType() != 'Bool':
        condition = g.op("Cast", condition, to_i=sym_help.cast_pytorch_to_onnx['Bool'])
    if self is None:
        condition = torch.onnx.symbolic_opset9.nonzero(g, condition)
        return sym_help._unbind_helper(g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs)
    return g.op("Where", condition, self, other)


@parse_args('v', 'i', 'none')
def log_softmax(g, input, dim, dtype=None):
    # PyTorch dim and ONNX axis have different meanings.
    # See Softmax comment for details.
    # TODO: remove this as onnx opset 11 spec allows negative axes
    input_dim = sym_help._get_tensor_rank(input)
    if input_dim is None:
        return _unimplemented("dim",
                              "ONNX and PyTorch use different strategies to split the input. "
                              "Input rank must be known at export time.")
    if dim < 0:
        dim = input_dim + dim
    is_transpose_required = (input_dim != dim + 1)
    # ONNX only supports log_softmax with dim = -1. Transpose must be added before and after log_softmax to support other cases.
    if is_transpose_required:
        axes = list(range(input_dim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        input = g.op("Transpose", input, perm_i=axes)
        dim = input_dim - 1
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        return_op = g.op("Cast", return_op, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    if is_transpose_required:
        return_op = g.op("Transpose", return_op, perm_i=axes)
    return return_op


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is', 'i', 'i', 'i', 'i', 'i')
def _convolution(g, input, weight, bias, stride, padding, dilation,
                 transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32):
    weight_size = sym_help._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        kernel_shape = None

    if kernel_shape is None or any([i is None for i in kernel_shape]):
        raise RuntimeError('Unsupported: ONNX export of convolution for kernel '
                           'of unknown shape.')

    args = [input, weight]
    # ONNX only supports 1D bias
    if not sym_help._is_none(bias) and sym_help._get_tensor_rank(bias) == 1:
        args.append(bias)

    kwargs = {"kernel_shape_i": weight_size[2:],
              "strides_i": stride,
              # NB: ONNX supports asymmetric padding, whereas PyTorch supports only
              # symmetric padding
              "pads_i": padding + padding,
              "dilations_i": dilation,
              "group_i": groups}

    if any(o != 0 for o in output_padding):
        # ONNX supports both output_shape and output_padding. they are equivalent expressive.
        # output_padding is more straightforward, so we use it here.
        # output_shape = stride * (input_shape - 1) + output_padding + kernel_shape - padding * 2
        assert transposed
        assert len(stride) == len(output_padding)
        kwargs["output_padding_i"] = output_padding

    n = g.op("ConvTranspose" if transposed else "Conv", *args, **kwargs)

    if not sym_help._is_none(bias) and sym_help._get_tensor_rank(bias) != 1:
        return g.op("Add", n, bias)
    else:
        return n


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv1d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv2d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv3d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose1d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose2d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose3d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None, None)


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    sym_help.assert_training_mode(training, "batch_norm")
    batch_size = sym_help._get_tensor_dim_size(input, 0)
    channel_size = sym_help._get_tensor_dim_size(input, 1)

    if weight is None or sym_help._is_none(weight):
        if channel_size is None:
            raise RuntimeError('Unsupported: ONNX export of batch_norm for unknown '
                               'channel size.')
        weight_value = torch.tensor([1.] * channel_size).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        if channel_size is None:
            raise RuntimeError('Unsupported: ONNX export of batch_norm for unknown '
                               'channel size.')
        bias_value = torch.tensor([0.] * channel_size).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)
    # If track_running_stats is set to False batch statistics are instead used during evaluation time
    if running_mean is None or sym_help._is_none(running_mean) or running_var is None or sym_help._is_none(running_var):
        assert batch_size is not None and channel_size is not None
        reshape_in = g.op("Reshape", input,
                          g.op("Constant", value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64)))
        trans_in = g.op('Transpose', reshape_in, perm_i=[0, 2, 1])
        running_var, running_mean = _var_mean(g, trans_in,
                                              g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64)),
                                              False, False)
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               outputs=1 if not sym_help._training_mode else 5)
    if not sym_help._training_mode:
        return out
    else:
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        return res


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, weight, bias, normalized_shape_i=normalized_shape,
                    eps_f=eps, cudnn_enable_i=cudnn_enable, operator_s="layer_norm")

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = g.op("Constant", value_t=torch.tensor(2.))
    eps_cst = g.op("Constant", value_t=torch.tensor(eps))

    mean = g.op("ReduceMean", input, axes_i=axes)
    numerator = sub(g, input, mean)
    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, add(g, variance, eps_cst))

    layer_norm = g.op("Div", numerator, denominator)

    if not (weight is None or sym_help._is_none(weight)):
        layer_norm = mul(g, layer_norm, weight)
    if not (bias is None or sym_help._is_none(bias)):
        layer_norm = add(g, layer_norm, bias)

    return layer_norm


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def instance_norm(g, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled):
    channel_size = sym_help._get_tensor_dim_size(input, 1)
    if weight is None or sym_help._is_none(weight):
        if channel_size is None:
            raise RuntimeError('Unsupported: ONNX export of instance_norm for unknown '
                               'channel size.')
        weight_value = torch.tensor([1.] * channel_size).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        if channel_size is None:
            raise RuntimeError('Unsupported: ONNX export of instance_norm for unknown '
                               'channel size.')
        bias_value = torch.tensor([0.] * channel_size).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)
    return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)


@parse_args('v', 'i', 'i', 'i')
def unfold(g, input, dimension, size, step):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)
    sizes = sym_help._get_tensor_sizes(input)
    try:
        sizedim = sizes[dimension]
    except Exception:
        sizedim = None
    if sizedim is not None:
        low_indices = range(0, sizedim, step)
        hi_indices = range(size, sizedim + 1, step)
        stack = [sym_help._slice_helper(g, input, axes=[dimension], starts=[low], ends=[hi])
                 for low, hi in zip(low_indices, hi_indices)]
        ndim = len(sizes)
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))
        unsqueeze = [sym_help._unsqueeze_helper(g, g.op("Transpose", t, perm_i=perm), [dimension]) for t in stack]
        return g.op("Concat", *unsqueeze, axis_i=dimension)
    else:
        return _unimplemented("Unfold", "input size not accessible")


@parse_args('v', 't', 't', 't')
def elu(g, input, alpha, scale, input_scale):
    if scale and scale != 1.:
        return _unimplemented("scale", "does not support scale in Elu")
    if input_scale and input_scale != 1.:
        return _unimplemented("input_scale", "does not support input_scale in Elu")
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=sym_help._scalar(alpha))


def selu(g, input):
    return g.op("Selu", input)


@parse_args('v', 'i', 'v')
def index_select(g, self, dim, index):
    # In case of a scalar index, index_select returns a tensor with the same rank as the input.
    # To match this behavior in ONNX, we make index a 1D tensor so that the following gather
    # also produces a tensor with the same rank as the input.
    return sym_help._select_helper(g, self, dim, index)


def index_put(g, self, indices_list_value, values, accumulate):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        indices_list = sym_help._unpack_list(indices_list_value)
        args = [self] + indices_list + [values, accumulate]
        return g.op("ATen", *args, operator_s='index_put')
    else:
        sym_help._onnx_opset_unsupported('index_put', 9, 11)


def index_fill(g, self, dim, index, value):
    dim_value = sym_help._parse_arg(dim, 'i')
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, value, dim_i=dim_value, operator_s="index_fill")
    expanded_index_shape, expanded_index = sym_help._index_fill_reshape_helper(g, self, dim, index)
    value = sym_help._maybe_get_scalar(value)
    value = sym_help._if_scalar_type_as(g, value, self)
    expanded_value = expand(g, value, expanded_index_shape, None)

    return scatter(g, self, dim, expanded_index, expanded_value)


def index_copy(g, self, dim, index, source):
    dim_value = sym_help._parse_arg(dim, 'i')
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, source, dim_i=dim_value, operator_s="index_copy")
    expanded_index_shape, expanded_index = sym_help._index_fill_reshape_helper(g, self, dim, index)
    return scatter(g, self, dim, expanded_index, source)


def type_as(g, self, other):
    self_dtype = sym_help._try_get_scalar_type(self)
    other_dtype = sym_help._try_get_scalar_type(other)
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    if other_dtype is not None:
        return g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[other_dtype])
    else:
        if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
            # We don't know the type of other, bail by emitting ATen
            return g.op("ATen", self, other, operator_s="type_as")
        else:
            raise RuntimeError('Unsupported: ONNX export of type_as for tensor '
                               'of unknown dtype.')


@parse_args('v', 'v', 'i', 'f')
def cosine_similarity(g, x1, x2, dim, eps):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", x1, x2, dim_i=dim, eps_f=eps, operator_s="cosine_similarity")
    else:
        return sym_help._onnx_unsupported('cosine_similarity')


# ignore clone operators that are inserted by PyTorch autograd
def clone(g, input, unused_memory_format):
    return input


def abs(g, self):
    return g.op("Abs", self)


def log(g, self):
    return g.op("Log", self)


def log1p(g, self):
    return log(g, add(g, sym_help._if_scalar_type_as(g, torch.ones(1), self), self))


def pow(g, self, exponent):
    f_dtype = self_dtype = self.type().scalarType()
    if not sym_help._is_fp(self):
        f_dtype = 'Float'
        self = g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[f_dtype])
    if not sym_help._is_fp(exponent):
        exponent = g.op("Cast", exponent, to_i=sym_help.cast_pytorch_to_onnx[f_dtype])
    pow = g.op("Pow", self, exponent)
    if self_dtype and self_dtype != f_dtype:
        pow = g.op("Cast", pow, to_i=sym_help.cast_pytorch_to_onnx[self_dtype])
    return pow


def clamp(g, self, min, max):
    # min or max may be None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax
    if sym_help._is_none(min):
        return clamp_max(g, self, max)
    elif sym_help._is_none(max):
        return clamp_min(g, self, min)
    else:
        min = _parse_arg(min, 'f')
        max = _parse_arg(max, 'f')
        return g.op("Clip", self, min_f=min, max_f=max)


@parse_args('v', 'f')
def clamp_min(g, self, min):
    return g.op("Clip", self, min_f=min)


@parse_args('v', 'f')
def clamp_max(g, self, max):
    return g.op("Clip", self, max_f=max)


# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
def max(g, self, dim_or_y=None, keepdim=None):
    # torch.max(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    # torch.max(input, other)
    if keepdim is None:
        return g.op("Max", self, dim_or_y)
    # torch.max(input, dim, keepdim)
    else:
        dim = sym_help._get_const(dim_or_y, 'i', 'dim')
        keepdim = sym_help._get_const(keepdim, 'i', 'keepdim')
        max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op('ArgMax', self, axis_i=dim, keepdims_i=keepdim)
        return max, indices


def min(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    # torch.min(input, other)
    if keepdim is None:
        return g.op("Min", self, dim_or_y)
    # torch.min(input, dim, keepdim)
    else:
        dim = sym_help._get_const(dim_or_y, 'i', 'dim')
        keepdim = sym_help._get_const(keepdim, 'i', 'keepdim')
        min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op('ArgMin', self, axis_i=dim, keepdims_i=keepdim)
        return min, indices


def exp(g, self):
    return g.op("Exp", self)


@parse_args('v', 'f', 'i')
def dropout(g, input, p, train):
    sym_help.assert_training_mode(train, "dropout")
    # in eval mode, dropout is non-op - if the node's train param is set to False, dropout is non-op
    if not sym_help._training_mode:
        return input
    warnings.warn("Dropout is a training op and should not be exported in inference mode. "
                  "Make sure to call eval() on the model, and to export it with param training=False.")
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


def _unsupported_dropout(name):
    @parse_args('v', 'f', 'i')
    def feature_dropout(g, input, p, train):
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        if train:
            return _unimplemented(name, "training mode")
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


@parse_args('v', 't', 'is', 'i')
def norm(g, self, p, dim, keepdim):
    if p == 1:
        f = _reduce_op_symbolic("ReduceL1")
    elif p == 2:
        f = _reduce_op_symbolic("ReduceL2")
    else:
        raise RuntimeError("ONNX export only p-norms with p of 1 or 2")
    return f(g, self, dim=dim, keepdim=keepdim)


@parse_args('v', 'v', 'v', 'i')
def conv_tbc(g, input, weight, bias, pad):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, weight, bias, operator_s="conv_tbc", pad_i=pad)
    else:
        return sym_help._onnx_unsupported('conv_tbc')


@parse_args('v', 'i', 'i')
def _unique(g, input, sorted, return_inverse):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, operator_s="_unique", sorted_i=sorted,
                    return_inverse_i=return_inverse, outputs=2)
    else:
        return sym_help._onnx_unsupported('_unique')


@parse_args('v', 'i', 'i', 'i')
def _unique2(g, input, sorted, return_inverse, return_counts):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, operator_s="_unique2", sorted_i=sorted,
                    return_inverse_i=return_inverse, return_counts_i=return_counts,
                    outputs=3)
    else:
        sym_help._onnx_opset_unsupported('_unique2', 9, 11)


for k, v in sym_help.cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = parse_args('v', 'i')(partial(sym_help._cast_func_template, v))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


def new_empty(g, self, sizes, dtype, layout, device, pin_memory=False):
    self_dtype = sym_help._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    return empty(g, sizes, dtype, layout, device, pin_memory)


def scalar_tensor(g, scalar, dtype, *options):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    scalar = g.op("Cast", scalar, to_i=sym_help.scalar_type_to_onnx[dtype])
    return scalar


def tensor(g, data, dtype=None, device=None, requires_grad=False):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if sym_help._is_packed_list(data):
        if dtype is None:
            dtype = sym_help._unpack_list(data)[0].type().scalarType()
            dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
        input_list = list()
        for t in sym_help._unpack_list(data):
            shape_reference = g.op("Constant", value_t=torch.LongTensor([1]))
            t = g.op("Reshape", t, shape_reference)
            t = g.op("Cast", t, to_i=sym_help.scalar_type_to_onnx[dtype])
            input_list.append(t)
        return g.op("Concat", *input_list, axis_i=0)
    else:
        if dtype is None:
            dtype = data.type().scalarType()
            dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    return g.op("Cast", data, to_i=sym_help.scalar_type_to_onnx[dtype])


@parse_args('v', 'i', 'v', 'v', 'v')
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device, layout and pin_memory in ONNX, so we ignore it
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def zeros_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


def new_zeros(g, self, sizes, dtype, layout, device, pin_memory=False):
    self_dtype = sym_help._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@parse_args('v', 'i', 'v', 'v', 'v')
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([1], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def ones_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([1], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    const_value = sym_help._maybe_get_const(value, 't')
    if sym_help._is_value(const_value):
        dtype = 6 if dtype is None else dtype
        tmp = zeros(g, sizes, dtype, layout, device)
        return add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = sym_help._get_const(dtype, 'i', 'dtype')
        dtype = 6 if dtype is None else dtype
        return g.op("ConstantOfShape", sizes,
                    value_t=torch.tensor([const_value], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


def full_like(g, input, fill_value, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    fill_value = sym_help._maybe_get_const(fill_value, 'f')
    if sym_help._is_value(fill_value):
        dtype = 6 if dtype is None else dtype
        tmp = zeros_like(g, input, dtype, layout, device)
        return add(g, tmp, fill_value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = sym_help._get_const(dtype, 'i', 'dtype')
        dtype = 6 if dtype is None else dtype
        shape = g.op("Shape", input)
        return g.op("ConstantOfShape", shape,
                    value_t=torch.tensor([fill_value], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


def new_full(g, self, size, fill_value, dtype, layout, device, pin_memory=False):
    self_dtype = sym_help._try_get_scalar_type(self)
    if dtype is None and self_dtype is not None:
        dtype = self_dtype
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    return full(g, size, fill_value, dtype, layout, device, pin_memory)


def eye(g, *args):
    if len(args) == 5:
        # aten::eye(n, dtype, layout, device, pin_memory)
        n, dtype, layout, device, pin_memory = args
        dim_size = sym_help._unsqueeze_helper(g, n, [0])
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    elif len(args) == 6:
        # aten::eye(n, m, dtype, layout, device, pin_memory)
        n, m, dtype, layout, device, pin_memory = args
        shape = g.op("Concat", sym_help._unsqueeze_helper(g, n, [0]), sym_help._unsqueeze_helper(g, m, [0]), axis_i=0)
        tensor = zeros(g, shape, dtype, layout, device)
        return g.op("EyeLike", tensor)
    else:
        raise NotImplementedError("Unknown aten::eye signature")


def slice(g, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        dim, start, end, step = args
        step = _parse_arg(step, 'i')
        if step != 1:
            raise RuntimeError("step!=1 is currently not supported")
        if start.node().kind() != 'onnx::Constant' or \
                end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant':
            if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX:
                raise RuntimeError('Unsupported: ONNX export of Slice with dynamic inputs. DynamicSlice '
                                   'is a deprecated experimental op. Please use statically allocated '
                                   'variables or export to a higher opset version.')
            else:
                start_unsqueezed = sym_help._unsqueeze_helper(g, start, [0])
                end_unsqueezed = sym_help._unsqueeze_helper(g, end, [0])
                dim_unsqueezed = sym_help._unsqueeze_helper(g, dim, [0])
                return g.op("DynamicSlice", self, start_unsqueezed, end_unsqueezed, dim_unsqueezed)
        else:
            start = _parse_arg(start, 'i')
            end = _parse_arg(end, 'i')
            dim = _parse_arg(dim, 'i')
            return sym_help._slice_helper(g, self, axes=[dim], starts=[start], ends=[end])
    elif len(args) == 3:
        # aten::slice(t[] l, int start, int end, int step) -> t[]
        start, end, step = args
        dim = 0
        start = _parse_arg(start, 'i')
        end = _parse_arg(end, 'i')
        return sym_help._slice_helper(g, self, axes=[dim], starts=[start], ends=[end])
    else:
        raise NotImplementedError("Unknown aten::slice signature")


@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    return g.op("Clip", self, min_f=min_val, max_f=max_val)


@parse_args('v')
def hardswish(g, self):
    input = g.op("Add", self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
    hardtanh_ = sym_help._hardtanh_helper(g, input,
                                          g.op('Constant', value_t=torch.tensor(0, dtype=torch.float)),
                                          g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
    hardtanh_ = g.op("Div", hardtanh_, g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
    return g.op("Mul", self, hardtanh_)

def alias(g, self):
    return self


@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    # Handle negative dim
    if dim < 0:
        rank = sym_help._get_tensor_rank(self)
        if rank is not None:
            warnings.warn("ONNX export unsqueeze with negative axis " + str(dim) +
                          " might cause the onnx model to be incorrect. " +
                          "Negative axis is not supported in ONNX. " +
                          "Axis is converted to " + str(dim + rank + 1) +
                          " based on input shape at export time. " +
                          "Passing an tensor of different rank in execution will be incorrect.")
            dim = dim + rank + 1
        else:
            return _unimplemented('unsqueeze', 'negative axis with unknown input rank')

    return sym_help._unsqueeze_helper(g, self, axes_i=[dim])


@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported for sort")
    self_sizes = sym_help._get_tensor_sizes(self)
    try:
        dim_size = self_sizes[dim]
    except Exception:
        dim_size = None

    if dim_size is None:
        return _unimplemented("Sort", "input size not accessible")

    return g.op("TopK", self, k_i=dim_size, axis_i=dim, outputs=2)


def numel(g, self):
    shape = g.op("Shape", self)
    return g.op("ReduceProd", shape, keepdims_i=0)


@parse_args('v', 'i', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")

    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


def to(g, self, *args):
    # ONNX doesn't have a concept of a device, so we ignore device casts
    if len(args) == 4:
        if args[0].type().isSubtypeOf(ListType.ofInts()):
            # aten::to(Tensor, Device, bool, bool, memory_format)
            return self
        else:
            dtype = sym_help._maybe_get_const(args[0], 'i')
            if sym_help._is_value(dtype):
                # aten::to(Tensor, Tensor, bool, bool, memory_format)
                other = args[0]
                dtype = other.type().scalarType()
                return g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[dtype])
            else:
                # aten::to(Tensor, ScalarType, bool, bool, memory_format)
                # memory_format is ignored
                return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        dtype = sym_help._get_const(args[1], 'i', 'dtype')
        # memory_format is ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) -> Tensor
        dtype = sym_help._get_const(args[0], 'i', 'dtype')
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) -> Tensor
        dtype = sym_help._get_const(args[0], 'i', 'dtype')
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    else:
        raise NotImplementedError("Unknown aten::to signature")


def repeat(g, self, repeats):
    dtype = 4  # int64
    shape_ = ones_like(g, repeats, dtype)
    self = g.op("Expand", self, shape_)
    return g.op("Tile", self, repeats)


@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    dims = sym_help._get_tensor_sizes(self)
    if len(dims) != 4:
        return _unimplemented("pixel_shuffle", "only support 4d input")
    if any([i is None for i in dims[1:]]):
        return _unimplemented("pixel_shuffle", "only support static input shape, except for batch size")
    output_channel = dims[1] // upscale_factor // upscale_factor
    after_view = view(g, self, g.op("Constant", value_t=torch.tensor([-1, output_channel, upscale_factor,
                                                                      upscale_factor, dims[2], dims[3]])))
    after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
    return view(g, after_transpose,
                g.op("Constant", value_t=torch.tensor([-1, output_channel, dims[2] * upscale_factor,
                                                       dims[3] * upscale_factor])))


def _generic_rnn(g, variant, input, initial_states, all_weights, has_biases,
                 num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None):

    warnings.warn("Exporting a model to ONNX with a batch_size other than 1, " +
                  "with a variable length with " + variant + " can cause an error " +
                  "when running the ONNX model with a different batch size. " +
                  "Make sure to save the model with a batch size of 1, " +
                  "or define the initial states (h0/c0) as inputs of the model. ")

    onnxActivations = ['Relu', 'Tanh', 'Sigmoid', 'Affine', 'LeakyRelu', 'ThresholdedRelu',
                       'ScaledTanh', 'HardSigmoid', 'Elu', 'Softsign', 'Softplus']
    variantToOnnxActivationMap = dict(zip([act_fun.lower() for act_fun in onnxActivations], onnxActivations))
    weights_per_layer = 4 if has_biases else 2
    # this means that projections are used inside LSTM, so need to tell user that it's not supported
    if variant == 'LSTM' and len(all_weights) != num_layers * weights_per_layer * (1 + bidirectional):
        return _unimplemented("LSTM", "LSTMs with projections")
    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [all_weights[i:i + weights_per_layer] for i in range(0, len(all_weights), weights_per_layer)]
    if batch_first:
        # batch, seq, feat -> seq, batch, feat
        input = g.op('Transpose', input, perm_i=[1, 0, 2])
    if dropout and train:
        return _unimplemented("RNN/GRU/LSTM", "dropout in training mode")

    if variant.startswith('RNN'):
        nonlinearity = variantToOnnxActivationMap[variant[4:].lower()]
        variant = 'RNN'

    w_hh = all_weights[1]
    hidden_size = sym_help._get_tensor_dim_size(w_hh, 1)
    if hidden_size is None:
        return _unimplemented("RNN/GRU/LSTM", "unknown hidden size")

    unidirectional = not bidirectional

    prev_output = input

    h_outs = []
    if variant == 'RNN' or variant == 'GRU':
        h0 = initial_states
    elif variant == 'LSTM':
        h0, c0 = initial_states
        c_outs = []

    sequence_lens = unused(g) if batch_sizes is None else batch_sizes

    if variant == 'GRU':
        # pytorch is reset, input, hidden
        # onnx is    input, reset, hidden
        reform_permutation = [(1, 2), (0, 1), (2, 3)]
    elif variant == 'LSTM':
        # pytorch is input, forget, cell, output.
        # onnx is    input, output, forget, cell.
        reform_permutation = [(0, 1), (3, 4), (1, 3)]

    def reform_weights(g, w, n, intervals):
        slices = [sym_help._slice_helper(g, w, axes=[0], starts=[x * n], ends=[y * n]) for x, y in intervals]
        return g.op('Concat', *slices, axis_i=0)

    def transform_weights_no_bias(layer_index):
        weights = layer_weights[layer_index]
        if variant == 'RNN':
            weight_ih, weight_hh = weights
        elif variant == 'GRU' or variant == 'LSTM':
            weight_ih, weight_hh = \
                [reform_weights(g, w, hidden_size, reform_permutation) for w in weights]
        return tuple(sym_help._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh))

    def transform_weights(layer_index):
        weights = layer_weights[layer_index]
        if variant == 'RNN':
            weight_ih, weight_hh, bias_ih, bias_hh = weights
        elif variant == 'GRU' or variant == 'LSTM':
            weight_ih, weight_hh, bias_ih, bias_hh = \
                [reform_weights(g, w, hidden_size, reform_permutation) for w in weights]
        bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)
        return tuple(sym_help._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh, bias_concat))

    def retrieve_state(x, start, end):
        return x if num_layers == 1 else sym_help._slice_helper(g, x, axes=[0], starts=[start], ends=[end])

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
                bias_concat = g.op('Concat', bias_f, bias_b, axis_i=0)
            else:
                weight_ih_f, weight_hh_f = transform_weights_no_bias(2 * i)
                weight_ih_b, weight_hh_b = transform_weights_no_bias(2 * i + 1)
                bias_concat = unused(g)

            weight_ih = g.op('Concat', weight_ih_f, weight_ih_b, axis_i=0)
            weight_hh = g.op('Concat', weight_hh_f, weight_hh_b, axis_i=0)

            state_indices = 2 * i, 2 * i + 2

        inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens]

        inputs.append(retrieve_state(h0, *state_indices))
        if variant == 'LSTM':
            inputs.append(retrieve_state(c0, *state_indices))

        extra_kwargs = {} if unidirectional else {'direction_s': 'bidirectional'}
        if variant == 'RNN':
            if bidirectional:
                activation = [nonlinearity, nonlinearity]
            else:
                activation = [nonlinearity]

            prev_output, h_out = g.op('RNN', *inputs, outputs=2,
                                      hidden_size_i=hidden_size,
                                      activations_s=activation,
                                      **extra_kwargs)
        elif variant == 'GRU':
            prev_output, h_out = g.op('GRU', *inputs, outputs=2,
                                      hidden_size_i=hidden_size,
                                      linear_before_reset_i=1,
                                      **extra_kwargs)
        elif variant == 'LSTM':
            prev_output, h_out, c_out = g.op('LSTM', *inputs, outputs=3,
                                             hidden_size_i=hidden_size,
                                             **extra_kwargs)

        if bidirectional:
            # The ONNX RNN/GRU/LSTM produce an output of dimensions
            #   seq_len, num_directions, batch, hidden_size
            # We have to convert to match pytorch's expected
            #   seq_len, batch, num_directions * hidden_size
            # by first moving num_directions before hidden_size with
            # Transpose, and then combining it with hidden_size
            # with Reshape.
            prev_output = g.op('Transpose', prev_output, perm_i=[0, 2, 1, 3])
            prev_output = g.op('Reshape', prev_output, g.op('Constant', value_t=torch.LongTensor([0, 0, -1])))
        else:
            prev_output = sym_help._squeeze_helper(g, prev_output, [1])

        h_outs.append(h_out)
        if variant == 'LSTM':
            c_outs.append(c_out)
    if batch_first:
        # seq, batch, num_directions * hidden_size -> batch, seq, num_directions * hidden_size
        prev_output = g.op('Transpose', prev_output, perm_i=[1, 0, 2])
    h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
    if variant == 'RNN' or variant == 'GRU':
        return prev_output, h_outs
    elif variant == 'LSTM':
        c_outs = c_out if num_layers == 1 else g.op('Concat', *c_outs, axis_i=0)
        return prev_output, h_outs, c_outs


@parse_args('v', 'v', 'v', 'i', 'i', 'f', 'i', 'i', 'i')
def _lstm_full(g, input, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    hidden, weight = sym_help._unpack_list(hidden_v), sym_help._unpack_list(weight_v)
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers,
                        dropout, train, bidirectional, batch_first)


@parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
def _lstm_packed(g, input, batch_sizes, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional):
    hidden, weight = sym_help._unpack_list(hidden_v), sym_help._unpack_list(weight_v)
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers,
                        dropout, train, bidirectional, batch_sizes=batch_sizes)


def lstm(g, *args):
    if sym_help._is_tensor_list(args[3]):
        return _lstm_packed(g, *args)
    else:
        return _lstm_full(g, *args)


def _one_hidden_rnn(kind):
    @parse_args('v', 'v', 'v', 'i', 'i', 'f', 'i', 'i', 'i')
    def _rnn_full(g, input, hidden, weight_v, has_biases, num_layers, dropout, train, bidirectional, batch_first):
        weight = sym_help._unpack_list(weight_v)
        return _generic_rnn(g, kind, input, hidden, weight, has_biases, num_layers,
                            dropout, train, bidirectional, batch_first)

    @parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
    def _rnn_packed(g, input, batch_sizes, hidden, weight_v, has_biases, num_layers, dropout, train, bidirectional):
        weight = sym_help._unpack_list(weight_v)
        return _generic_rnn(g, kind, input, hidden, weight, has_biases, num_layers,
                            dropout, train, bidirectional, batch_sizes=batch_sizes)

    def symbolic(g, *args):
        if sym_help._is_tensor_list(args[3]):
            return _rnn_packed(g, *args)
        else:
            return _rnn_full(g, *args)

    return symbolic


gru = _one_hidden_rnn('GRU')
rnn_tanh = _one_hidden_rnn('RNN_TANH')
rnn_relu = _one_hidden_rnn('RNN_RELU')


@parse_args('v', 'i')
def _dim_arange(g, like, dim):
    like_shape = g.op('Shape', like)
    stop = g.op("Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0)
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("_caffe2::Range", stop)
    else:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        return arange(g, stop, 4, None, None, None)


def detach(g, input):
    # Erase aten::detach nodes because ONNX is inference only
    return input


@parse_args('v', 'i')
def contiguous(g, input, memory_format):
    if memory_format > 2:  # allower values are any, preserve and contiguous_format
        raise RuntimeError("onnx memory_format support is not implemented")
    return input


@parse_args('v', 'v', 'i')
def _pack_padded_sequence(g, input, lengths, batch_first):
    # There currently is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.
    if batch_first:
        input = g.op('Transpose', input, perm_i=[1, 0, 2])
    if not lengths.type().isSubtypeOf(torch._C.TensorType.get()):
        raise RuntimeError("Lengths must be a Tensor for ONNX export")
    # We know it's a TensorType so this check is now safe.
    # It's really only necessary because those operators expand to something that
    # only works with int32 types in Caffe2...
    if lengths.type().scalarType() != 'Int':
        lengths = _cast_Int(g, lengths, False)  # type: ignore
    return g.op("prim::PackPadded", input, lengths, outputs=2)


@parse_args('v', 'v', 'i', 't', 'v')
def _pad_packed_sequence(g, data, batch_sizes, batch_first, padding_value, total_length):
    # Ignore total_length as it is not supported in _symbolic_pad_packed_sequence
    # It is only useful/used when training using data_parallel model, so
    # It shouldn't be relevant for ONNX anyway
    data, lengths = g.op("prim::PadPacked", data, batch_sizes, outputs=2)
    if batch_first:
        data = g.op('Transpose', data, perm_i=[1, 0, 2])
    return data, lengths


def randn(g, shapes, dtype, *options):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    shape = sym_help._maybe_get_const(shapes, "is")
    if sym_help._is_value(shape):
        shape_const = g.op("ConstantOfShape", shapes,
                           value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[6]))
        return g.op('RandomNormalLike', shape_const, dtype_i=sym_help.scalar_type_to_onnx[dtype])
    return g.op('RandomNormal', shape_i=shape)


def rand(g, shapes, dtype, *options):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    shape = sym_help._maybe_get_const(shapes, "is")
    if sym_help._is_value(shape):
        shape_const = g.op("ConstantOfShape", shapes,
                           value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[6]))
        return g.op('RandomUniformLike', shape_const, dtype_i=sym_help.scalar_type_to_onnx[dtype])
    return g.op('RandomUniform', shape_i=shape)


def randn_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    return g.op('RandomNormalLike', self, dtype_i=sym_help.scalar_type_to_onnx[dtype])


def rand_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    return g.op('RandomUniformLike', self, dtype_i=sym_help.scalar_type_to_onnx[dtype])


@parse_args('v', 'f', 'f', 'i', 'none')
def rrelu(g, input, lower, upper, training, generator):
    p = g.op('RandomUniformLike', input, high_f=upper, low_f=lower)
    return g.op('PRelu', input, p)


@parse_args('v')
def log_sigmoid(g, input):
    p = g.op('Sigmoid', input)
    return g.op('Log', p)


@parse_args('v')
def erf(g, input):
    return g.op('Erf', input)


@parse_args('v', 'i', 'i')
def flatten(g, input, start_dim, end_dim):
    dim = sym_help._get_tensor_rank(input)
    if dim is None:
        return _unimplemented("dim",
                              "ONNX and PyTorch use different strategies to split the input. "
                              "Input rank must be known at export time.")

    # TODO: remove this as onnx opset 11 spec allows negative axes
    if end_dim < 0 :
        end_dim = dim + end_dim
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1 and end_dim == dim - 1 :
        return g.op("Flatten", input, axis_i=start_dim)
    if start_dim == 0 and end_dim == dim - 2 :
        return g.op("Flatten", input, axis_i=end_dim + 1)

    return sym_help._flatten_helper(g, input, start_dim, end_dim, dim)

# Emitted from `torch.nonzero(x, as_tuple=False)`
@parse_args('v')
def nonzero(g, input):
    return t(g, g.op('NonZero', input))


# Emitted from `torch.nonzero(x, as_tuple=True)`
def nonzero_numpy(g, input, _outputs=None):
    return unbind(g, nonzero(g, input), 1, _outputs=_outputs)


@parse_args('v')
def isnan(g, input):
    output = g.op('IsNaN', input)
    return output


@parse_args('v', 'i', 'i', 'i')
def narrow(g, input, dim, start, length):
    return sym_help._slice_helper(g, input, axes=[dim], starts=[start], ends=[start + length])


def argmax(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        flattened = reshape(g, input, g.op("Constant", value_t=torch.tensor([-1])))
        return g.op('ArgMax', flattened, axis_i=0, keepdims_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMax', input, axis_i=dim, keepdims_i=keepdim)


def argmin(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        flattened = reshape(g, input, g.op("Constant", value_t=torch.tensor([-1])))
        return g.op('ArgMin', flattened, axis_i=0, keepdims_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMin', input, axis_i=dim, keepdims_i=keepdim)


@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    src_type = src.type().scalarType()
    src = sym_help._maybe_get_scalar(src)
    if sym_help._is_value(src):
        return g.op("Scatter", self, index, src, axis_i=dim)
    else:
        # Check if scalar 'src' has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if self.type().scalarType() != src_type:
            src = g.op("Cast", src, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
        return g.op("Scatter", self, index, expand_as(g, src, index), axis_i=dim)


@parse_args('v', 'i', 'v', 'v')
def scatter_add(g, self, dim, index, src):
    dtype = sym_help._try_get_scalar_type(self)
    if dtype is None:
        return _unimplemented("scatter_add", "input dtype not accessible")
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    dtype = sym_help.scalar_type_to_pytorch_type[dtype]
    sizes = sym_help._get_tensor_sizes(self, allow_nonstatic=False)
    if sizes:
        to_add = g.op("Constant", value_t=torch.zeros(sizes, dtype=dtype))
    else:
        to_add = zeros_like(self, dtype)
    to_add = sym_help._scatter_helper(g, to_add, dim, index, src)
    return add(g, self, to_add)


def log2(g, self):
    _ln2 = 0.693147180559945309
    return g.op('Div', log(g, self), g.op('Constant', value_t=torch.Tensor([_ln2])))


def prim_shape(g, self):
    return g.op('Shape', self)

def prim_max(g, self, other):
    return g.op('Max', self, other)

def prim_data(g, self):
    return self

def is_floating_point(g, self):
    if sym_help._is_fp(self):
        return g.op("Constant", value_t=torch.BoolTensor([1]))
    return g.op("Constant", value_t=torch.BoolTensor([0]))


def __isnot_(g, self, other):
    if sym_help._is_none(other):
        if sym_help._is_none(self):
            return g.op("Constant", value_t=torch.BoolTensor([0]))
        return g.op("Constant", value_t=torch.BoolTensor([1]))
    return ne(g, self, other)


# exists to refine the type of the Value
# if x is an optional Tensor, unchecked_cast will cast
# x to Tensor, so the rest of the graph knows that x is a Tensor
# this doesn't do anything in runtime and is a noop in ONNX
def prim_unchecked_cast(g, self):
    return self


def prim_dtype(g, self):
    dtype = sym_help._try_get_scalar_type(self)
    if dtype is None:
        dtype = "Float"
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    return g.op("Constant", value_t=torch.tensor(dtype))


# tolist is currently supported only for 1D input tensors.
# dim_val and elem_ty_val represent dimension and type annotations
# that need to match dimension and type of the input tensor.
def prim_tolist(g, input, dim_val, elem_ty_val):
    dim = sym_help._maybe_get_const(dim_val, 'i')
    if dim > 1:
        return _unimplemented("prim_tolist", "dim_val > 1")
    return input


@parse_args('v', 'i')
def one_hot(g, self, num_classes):
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    depth = g.op("Constant", value_t=torch.LongTensor([num_classes]))
    return g.op("OneHot", self, depth, values, axis_i=-1)


@parse_args('v', 'i', 'v', 'v')
def gather(g, self, dim, index, sparse_grad=False):
    if sym_help._maybe_get_const(sparse_grad, 'i'):
        return _unimplemented("gather", "sparse_grad == True")
    # NOTE: This workaround is needed since GatherElement is only supported
    #       since opset 11, and Gather in ONNX is not the same as torch.gather.
    dtype = self.type().scalarType()
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    depth = size(g, self, g.op("Constant", value_t=torch.LongTensor([dim])))
    index = g.op("Cast", g.op("OneHot", index, depth, values, axis_i=dim), to_i=sym_help.cast_pytorch_to_onnx[dtype])
    mul = g.op("Mul", sym_help._unsqueeze_helper(g, self, [dim + 1]), index)
    return sym_help._reducesum_helper(g, mul, axes_i=[dim], keepdims_i=0)


@parse_args('v', 'is', 'b', 'i')
def _var_mean(g, input, dim, unbiased, keepdim):
    if dim is None:
        mean = g.op("ReduceMean", input, keepdims_i=0)
        t_mean = mean
        num_elements = numel(g, input)
    else:
        mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
        t_mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=1)
        redudced_dims = g.op("Shape", input)
        # dim could contain one or multiple dimensions
        redudced_dims = g.op("Gather", redudced_dims, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0)
        num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
    sub_v = g.op("Sub", input, t_mean)
    sqr_sub = g.op("Mul", sub_v, sub_v)
    keepdim_mean = 0 if dim is None else keepdim
    var = g.op("ReduceMean", sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
    # Correct bias in calculating variance, by dividing it over (N - 1) instead on N
    if unbiased:
        num_elements = g.op("Cast", num_elements, to_i=sym_help.cast_pytorch_to_onnx['Float'])
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.float))
        mul = g.op("Mul", var, num_elements)
        var = g.op("Div", mul, g.op("Sub", num_elements, one))
    return var, mean


# Since position of optional arguments can change for std, this is a hack to find if first argument
# is 'dim' or 'unbiased'. As shown below, 'dim' argument could be listed before 'unbiased' :
# at::std(input, unbiased)
# at::std(input, dim, unbiased, keepdim)
def std(g, input, *args):
    if len(args) == 3:
        var, _ = _var_mean(g, input, *args)
    else:
        var, _ = _var_mean(g, input, None, args[0], None)
    return g.op("Sqrt", var)


# Since position of optional arguments can change for var, this is a hack to find if first argument
# is 'dim' or 'unbiased'. As shown below, 'dim' argument could be listed before 'unbiased' :
# at::var(input, unbiased)
# at::var(input, dim, unbiased, keepdim)
def var(g, input, *args):
    if len(args) == 3:
        var, _ = _var_mean(g, input, *args)
    else:
        var, _ = _var_mean(g, input, None, args[0], None)
    return var


# Since position of optional arguments can change for var_mean, this is a hack to find if first argument
# is 'dim' or 'unbiased'. As shown below, 'dim' argument could be listed before 'unbiased' :
# at::var_mean(input, unbiased)
# at::var_mean(input, dim, unbiased, keepdim)
def var_mean(g, input, *args):
    if len(args) == 3:
        var, mean = _var_mean(g, input, *args)
    else:
        var, mean = _var_mean(g, input, None, args[0], None)
    return var, mean


# Since position of optional arguments can change for std_mean, this is a hack to find if first argument
# is 'dim' or 'unbiased'. As shown below, 'dim' argument could be listed before 'unbiased' :
# at::std_mean(input, unbiased)
# at::std_mean(input, dim, unbiased, keepdim)
def std_mean(g, input, *args):
    if len(args) == 3:
        var, mean = _var_mean(g, input, *args)
    else:
        var, mean = _var_mean(g, input, None, args[0], None)
    return g.op("Sqrt", var), mean


@parse_args('v', 'is', 'i')
def logsumexp(g, input, dim, keepdim):
    return g.op('ReduceLogSumExp', input, axes_i=dim, keepdims_i=keepdim)


def arange(g, *args):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", *args, operator_s="arange")

    def _get_arange_dtype(dtype):
        dtype = sym_help._maybe_get_const(dtype, 'i')
        if sym_help._is_value(dtype):
            dtype = 4  # default to int64
        return dtype

    if len(args) == 2:
        # aten::arange(Scalar end, Tensor out)
        end = sym_help._unsqueeze_helper(g, args[0], [0])
        dtype = 4  # default to int64
        arange_tensor = sym_help._squeeze_helper(g, nonzero(g, ones(g, end, dtype, None, None)), [1])
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 4:
        # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
        dtype = 4  # default to int64
        step = sym_help._unsqueeze_helper(g, args[2], [0])
        end = sym_help._unsqueeze_helper(g, args[1], [0])
        start = sym_help._unsqueeze_helper(g, args[0], [0])
        range_tensor = g.op("Div", g.op("Sub", end, start), step)
        arange_tensor = sym_help._squeeze_helper(g, nonzero(g, ones(g, range_tensor, None, None, None)), [1])
        arange_tensor = g.op("Add", g.op("Mul", arange_tensor, step), start)
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[1])
        end = sym_help._unsqueeze_helper(g, args[0], [0])
        arange_tensor = sym_help._squeeze_helper(g, nonzero(g, ones(g, end, dtype, *(args[2:]))), [1])
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        end = sym_help._unsqueeze_helper(g, args[1], [0])
        start = sym_help._unsqueeze_helper(g, args[0], [0])
        range_tensor = g.op("Sub", end, start)
        arange_tensor = g.op("Add", sym_help._squeeze_helper(g, nonzero(g, ones(g, range_tensor, dtype, *(args[3:]))), [1]), start)
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[3])
        step = sym_help._unsqueeze_helper(g, args[2], [0])
        end = sym_help._unsqueeze_helper(g, args[1], [0])
        start = sym_help._unsqueeze_helper(g, args[0], [0])
        range_tensor = g.op("Div", g.op("Sub", end, start), step)
        arange_tensor = sym_help._squeeze_helper(g, nonzero(g, ones(g, range_tensor, dtype, *(args[4:]))), [1])
        arange_tensor = g.op("Add", g.op("Mul", arange_tensor, step), start)
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    else:
        raise NotImplementedError("Unknown aten::arange signature taking " + str(len(args)) + " arguments.")


def masked_fill(g, self, mask, value):
    mask = _cast_Bool(g, mask, False)  # type: ignore
    value = sym_help._maybe_get_scalar(value)
    return g.op('Where', mask, sym_help._if_scalar_type_as(g, value, self), self)


def index(g, self, index):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, operator_s="index")

    if sym_help._is_packed_list(index):
        indices = sym_help._unpack_list(index)
    else:
        indices = [index]

    def try_mask_to_index(index):
        if not sym_help._is_none(index) and (index.type().scalarType() == "Byte" or index.type().scalarType() == "Bool"):
            if sym_help._export_onnx_opset_version < 9:
                raise RuntimeError("Exporting masked indices are only supported after ONNX opset 9.")
            warnings.warn("Exporting aten::index operator with indices of type Byte. "
                          "Only 1-D indices are supported. In any other case, "
                          "this will produce an incorrect ONNX graph.")
            index = sym_help._squeeze_helper(g, nonzero(g, index), [1])
        return index

    indices = [try_mask_to_index(idx) for idx in indices]
    if len(indices) == 1:
        return sym_help._select_helper(g, self, 0, indices[0], apply_reshape=False)
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
        adv_idx_indices = [i for i, idx in enumerate(indices) if not sym_help._is_none(idx)]

        if len(adv_idx_indices) == 0:
            return self
        elif len(adv_idx_indices) == 1:
            return index_select(g, self, adv_idx_indices[0], indices[adv_idx_indices[0]])
        else:
            rank = sym_help._get_tensor_rank(self)
            if rank is None:
                raise NotImplementedError("Unsupported aten::index operator of advanced indexing on tensor of unknown rank, " +
                                          "try turning on shape and type propagate during export: " +
                                          "torch.onnx._export(..., propagate=True).")
            # TODO: If indexing is supported natively in ONNX in future opsets,
            #       update the warning to recommend exporting with higher opset version.
            warnings.warn("Exporting aten::index operator of advanced indexing in opset " +
                          str(sym_help._export_onnx_opset_version) +
                          " is achieved by combination of multiple ONNX operators, " +
                          "including Reshape, Transpose, Concat, and Gather. " +
                          "If indices include negative values, the exported graph will produce incorrect results.")
            adv_idx_count = len(adv_idx_indices)
            shape_tensor = _shape_as_tensor(g, self)
            dim_tensor_list = [
                g.op("Gather", shape_tensor, g.op("Constant", value_t=torch.LongTensor([dim])), axis_i=0) for dim in range(rank)
            ]

            self = g.op("Transpose", self, perm_i=adv_idx_indices + [i for i in range(rank) if i not in adv_idx_indices])
            self = g.op("Flatten", self, axis_i=adv_idx_count)

            # Note that tensor indices will be broadcasted while accumulating. Thus we get the final subarray shape as well.
            cum_adv_index = indices[adv_idx_indices[-1]]
            multiplier = dim_tensor_list[adv_idx_indices[-1]]
            for i in range(adv_idx_count - 2, -1, -1):
                adv_index = g.op("Mul", indices[adv_idx_indices[i]], multiplier)
                cum_adv_index = g.op("Add", cum_adv_index, adv_index)
                multiplier = g.op("Mul", multiplier, dim_tensor_list[adv_idx_indices[i]])

            # perform gather
            self = index_select(g, self, 0, cum_adv_index)

            cum_adv_index_shape_tensor = _shape_as_tensor(g, cum_adv_index)
            # check if all advanced indices are consecutive.
            # Refer to https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
            # to understand how the subarray position is decided.
            if adv_idx_indices == list(range(adv_idx_indices[0], adv_idx_indices[-1] + 1)):
                # unfold regular index axes
                folded_adv_idx_shape_list = [g.op("Constant", value_t=torch.LongTensor([-1]))]  \
                    + [dim_tensor_list[i] for i in range(rank) if i not in adv_idx_indices]
                folded_adv_idx_shape = g.op("Concat", *folded_adv_idx_shape_list, axis_i=0)
                self = g.op("Reshape", self, folded_adv_idx_shape)

                # Transpose folded advanced indexed axis to its original location.
                adv_idx_permute = list(range(1, adv_idx_indices[0] + 1))                    \
                    + [0] + list(range(adv_idx_indices[0] + 1, rank - adv_idx_count + 1))
                self = g.op("Transpose", self, perm_i=adv_idx_permute)

                # unfold advanced index axes
                final_shape_list = [dim_tensor_list[i] for i in range(adv_idx_indices[0])]                      \
                    + [cum_adv_index_shape_tensor]                                                              \
                    + [dim_tensor_list[i] for i in range(adv_idx_indices[0], rank) if i not in adv_idx_indices]
                final_shape = g.op("Concat", *final_shape_list, axis_i=0)
            else:
                final_shape = g.op(
                    "Concat",
                    cum_adv_index_shape_tensor,
                    *[dim_tensor_list[i] for i in range(rank) if i not in adv_idx_indices],
                    axis_i=0)

            return g.op("Reshape", self, final_shape)


@parse_args('v', 'is', 'i')
def frobenius_norm(g, self, dim=None, keepdim=False):
    sqr = g.op('Mul', self, self)
    sumsqr = sym_help._reducesum_helper(g, sqr, axes_i=dim, keepdims_i=keepdim)
    return g.op('Sqrt', sumsqr)


@parse_args('v', 'i', 'b', 'v')
def multinomial(g, input, num_samples, replacement=False, generator=None):
    if generator is not None and not sym_help._is_none(generator):
        _unimplemented("Multinomial", "generator is not supported for multinomial")
    if not replacement and num_samples > 1:
        _unimplemented("Multinomial", "replacement=False when num_samples > 1 is not supported for multinomial")

    log_input = log(g, input)
    return g.op("Multinomial", log_input,
                dtype_i=sym_help.cast_pytorch_to_onnx['Long'],
                sample_size_i=num_samples)


def baddbmm(g, self, batch1, batch2, beta, alpha):
    dtype = self.type().scalarType()
    batch_mul = matmul(g, batch1, batch2)
    mul_a = mul(g, batch_mul, g.op("Cast", alpha, to_i=sym_help.cast_pytorch_to_onnx[dtype]))
    mul_b = mul(g, self, g.op("Cast", beta, to_i=sym_help.cast_pytorch_to_onnx[dtype]))
    return add(g, mul_a, mul_b)


def meshgrid(g, tensor_list):
    tensors = [view(g, t, g.op("Constant", value_t=torch.LongTensor([-1]))) for t in sym_help._unpack_list(tensor_list)]
    tensors_shape = [g.op("Shape", t) for t in tensors]
    out_shape = g.op("Concat", *tensors_shape, axis_i=0)
    out = []
    for i, t in enumerate(tensors):
        shape_i = [g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))] * len(tensors)
        shape_i[i] = tensors_shape[i]
        t_reshaped = _reshape_from_tensor(g, t, g.op("Concat", *shape_i, axis_i=0))
        out.append(g.op("Expand", t_reshaped, out_shape))
    return g.op("prim::ListConstruct", *out)


def remainder(g, input, other):
    div = g.op("Div", input, other)
    if sym_help._is_fp(input) or sym_help._is_fp(other):
        div = g.op("Floor", div)
    quo = g.op("Mul", div, other)
    return g.op("Sub", input, quo)


def gelu(g, self):
    _sqrt2 = 1.4142135623730951
    erf = g.op('Erf', g.op('Div', self, torch.tensor(_sqrt2, dtype=torch.double)))
    erf_plusone = add(g, erf, g.op('Constant', value_t=torch.tensor(1, dtype=torch.double)))
    return mul(g, mul(g, self, erf_plusone), g.op('Constant', value_t=torch.tensor(0.5, dtype=torch.double)))

@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, weight, bias, num_groups_i=num_groups,
                    eps_f=eps, cudnn_enabled_i=cudnn_enabled, operator_s="group_norm")

    channel_size = sym_help._get_tensor_dim_size(input, 1)
    if channel_size is not None:
        assert channel_size % num_groups == 0
    input_rank = sym_help._get_tensor_rank(input)
    if input_rank is None:
        return _unimplemented("group_norm", "unknown input rank")
    # 0 in the shape list keeps dimension value unchanged.
    shape = [0, num_groups, -1]
    input_reshaped = g.op('Reshape', input, g.op('Constant', value_t=torch.LongTensor(shape)))

    # C is always divisible by num_groups
    # Due to shape difference. we need to apply weight and bias after
    # instance norm computation and reshape
    weight_ = g.op("Constant", value_t=torch.tensor([1.] * num_groups).type(
        'torch.' + input.type().scalarType() + 'Tensor'))
    bias_ = g.op("Constant", value_t=torch.tensor([0.] * num_groups).type(
        'torch.' + input.type().scalarType() + 'Tensor'))

    norm_reshaped = g.op("InstanceNormalization", input_reshaped, weight_, bias_, epsilon_f=eps)
    norm = g.op('Reshape', norm_reshaped, g.op("Shape", input))

    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor([1.]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().mustBeNone():
        bias_value = torch.tensor([0.]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    axes = list(range(1, input_rank - 1))
    return add(g, mul(g, norm, sym_help._unsqueeze_helper(g, weight, axes)), sym_help._unsqueeze_helper(g, bias, axes))


@parse_args('v', 'v', 'i')
def _weight_norm(g, weight_v, weight_g, dim):
    rank = sym_help._get_tensor_rank(weight_v)
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
    elif sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", weight_v, weight_g, dim_i=dim, operator_s="_weight_norm")
    else:
        raise RuntimeError('Unsupported: ONNX export of _weight_norm for tensor '
                           'of unknown rank.')


def dim(g, self):
    '''Implement the dim functionality available for a pytorch tensor in ONNX'''
    # ONNX does not support dim directly in this opset so we can use 2 ops to get the info
    shape = g.op('Shape', self)
    return g.op('Size', shape)


def __getitem_(g, self, i):
    return select(g, self, g.op("Constant", value_t=torch.tensor([0])), i)


def take(g, self, index):
    self_flattened = g.op('Reshape', self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
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


@parse_args('v', 'v', 'i', 'b')
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
        return sym_help._reducesum_helper(g, output, keepdims_i=0)
    else:
        return sym_help._onnx_unsupported("kl_div with reduction other than none, mean, or sum.")


@parse_args('v', 'v', 'is', 'i')
def as_strided(g, self, sizes, strides, offset=None):
    sizes = sym_help._maybe_get_const(sizes, 'is')
    rank = len(strides)
    self_1d = g.op("Reshape", self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
    ind: Optional[torch.Tensor]
    if not sym_help._is_value(sizes):
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
            size = select(g, sizes, g.op("Constant", value_t=torch.tensor([0])), g.op("Constant", value_t=torch.tensor(i)))
            tmp_ind = g.op("Reshape", arange(g, size, 4, None, None, None), g.op("Constant", value_t=torch.tensor(r_size)))
            tmp_ind = g.op("Mul", tmp_ind, g.op("Constant", value_t=torch.tensor([stride])))
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
    return g.op("Cast", div, to_i=sym_help.cast_pytorch_to_onnx['Long'])
