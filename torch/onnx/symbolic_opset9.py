from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
import torch.onnx.symbolic_caffe2 as sym_caffe2

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
    return g.op("Div", self, other)


def reciprocal(g, self):
    return g.op("Div", torch.ones(1), self)


@parse_args('v', 'i')
def cat(g, tensor_list, dim):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@parse_args('v', 'i')
def stack(g, tensor_list, dim):
    unsqueezed = [g.op("Unsqueeze", t, axes_i=[dim]) for t in sym_help._unpack_list(tensor_list)]
    return g.op("Concat", *unsqueezed, axis_i=dim)


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
    return g.op("Gemm", mat1, mat2, self, beta_f=sym_help._scalar(beta), alpha_f=sym_help._scalar(alpha))


def neg(g, self):
    return g.op("Neg", self)


def sqrt(g, self):
    return g.op("Sqrt", self)


def rsqrt(g, self):
    return div(g, sym_help._if_scalar_type_as(g, torch.ones(1), self), sqrt(g, self))


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


def _reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
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
    if dtype.node().kind() != 'prim::Constant':
        return _unimplemented(name, "dtype")
    return g.op("ATen", input, operator_s="cumsum", dim_i=dim)


def _sample_dirichlet(g, self, generator):
    if not sym_help._is_none(generator):
        return _unimplemented('_sample_dirichlet',
                              'We are not able to export generator')
    return g.op("ATen", self, operator_s="_sample_dirichlet")


def _standard_gamma(g, self, generator):
    if not sym_help._is_none(generator):
        return _unimplemented('_standard_gamma',
                              'We are not able to export generator')
    return g.op("ATen", self, operator_s="_standard_gamma")


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def expand(g, self, size, implicit):
    size = sym_help._maybe_get_const(size, 'is')
    if not sym_help._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Expand", self, size)


def expand_as(g, self, other):
    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op("Gather", weight, indices)


@parse_args('v', 'v', 'v', 'i', 'i', 'i', 'v')
def embedding_bag(g,
                  embedding_matrix,
                  indices,
                  offsets,
                  scale_grad_by_freq,
                  mode,
                  sparse,
                  per_sample_weights):
    if not sym_help._is_none(per_sample_weights):
        raise RuntimeError('Unsupported: ONNX export of embedding_bag '
                           'with per_sample_weights')
    return g.op("ATen",
                embedding_matrix,
                indices,
                offsets,
                operator_s="embedding_bag",
                outputs=4,
                scale_grad_by_freq_i=scale_grad_by_freq,
                mode_i=mode,
                sparse_i=sparse)


def size(g, self, dim):
    if sym_help._maybe_get_const(dim, 'i') < 0:
        rank = self.type().dim()
        if rank:
            dim = sym_help._maybe_get_const(dim, 'i') + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    return sym_help._size_helper(g, self, dim)


@parse_args('v', 'i', 'i')
def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    if self.isCompleteTensor():
        axes = list(range(self.type().dim()))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return g.op("Transpose", self, perm_i=axes)
    else:
        # if we don't have dim information we cannot
        # output a permute so use ATen instead
        return g.op("ATen", self, operator_s="transpose", dim0_i=dim0, dim1_i=dim1)


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
        if self.isCompleteTensor():
            self_sizes = self.type().sizes()
            if self_sizes and len(size) == 2 and self_sizes[0] == size[0]:
                return g.op("Flatten", self, axis_i=1)
        shape = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Reshape", self, shape)


def prim_ConstantSplit(g, self, split_size, dim):
    size = self.type().sizes()[dim]
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
    split_size = (self.type().sizes()[dim] + chunks - 1) // chunks
    return prim_ConstantSplit(g, self, split_size, dim)


@parse_args('v', 'i', 'i')
def split(g, self, split_size, dim):
    size = self.type().sizes()[dim]
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=1)


@parse_args('v', 'is', 'i')
def split_with_sizes(g, self, split_sizes, dim):
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=1)


@parse_args('v', 'i')
def unbind(g, self, dim=0):
    # NOTE: This conversion of this node is handled in onnx peephole pass.
    # Due to that an additional Squeeze node needs to be inserted for each output from unbind.
    return g.op("aten::unbind", self, axis_i=dim)


@parse_args('v', 'i', 'v')
def select(g, self, dim, index):
    index = sym_help._maybe_get_scalar(index)
    if (not sym_help._is_value(index)) and (index < 0):
        if index == -1:
            end_index = 9223372036854775807
        else:
            end_index = index + 1
        slice_node = sym_help._slice_helper(g, self, axes=[dim], starts=[index], ends=[end_index])
        return g.op("Squeeze", slice_node, axes_i=[dim])
    else:
        return g.op("Gather", self, index, axis_i=dim)


def squeeze(g, self, dim=None):
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [sym_help._get_const(dim, 'i', 'dim')]
        # Handle negative dims
        for i, dim in enumerate(dims):
            if dim < 0:
                rank = self.type().dim()
                if rank:
                    warnings.warn("ONNX export squeeze with negative axis " + str(dim) +
                                  " might cause the onnx model to be incorrect. " +
                                  "Negative axis is not supported in ONNX. " +
                                  "Axis is converted to " + str(dim + rank) +
                                  " based on input shape at export time. " +
                                  "Passing an tensor of different rank in execution will be incorrect.")
                    dims[i] += rank
                else:
                    return _unimplemented('squeeze', 'negative axis with unknown input rank')

    return g.op("Squeeze", self, axes_i=dims)


def prelu(g, self, weight):
    if self.isCompleteTensor():
        self_sizes = self.type().sizes()
        if self_sizes and len(self_sizes) > 2:
            weight = g.op("Unsqueeze", weight, axes_i=list(range(1, len(self_sizes) - 1)))
    return g.op("PRelu", self, weight)


def relu(g, input):
    if input in sym_help._quantized_ops:
        return sym_caffe2.relu(g, input)

    return g.op("Relu", input)


def ceil(g, input):
    return g.op("Ceil", input)


def floor(g, input):
    return g.op("Floor", input)


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
    assert input.type().sizes()[dim] % 2 == 0

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
    # So use softmax when dim and axis both equal to ndim - 1
    # otherwise compute softmax using a subgraph with other operators
    input_dim = input.type().dim()
    if input_dim:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        if dim < 0:
            dim = input_dim + dim
        if input_dim == dim + 1:
            softmax = g.op('Softmax', input, axis_i=dim)
            if dtype and dtype.node().kind() != 'prim::Constant':
                parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
                softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
            return softmax

    exp = g.op('Exp', input)
    sum = g.op('ReduceSum', exp, axes_i=[dim])
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
    dim = input.type().sizes()[-len(padding):]
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
        if ceil_mode and not input.isCompleteTensor():
            return _unimplemented(name, "input size not accessible")
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
        if ceil_mode and not input.isCompleteTensor():
            return _unimplemented(name, "input size not accessible")
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
    @parse_args('v', 'is')
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
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        if not input.isCompleteTensor():
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return _unimplemented(name, 'input size not accessible')
        dim = input.type().sizes()[2:]
        # verify if output size % input size = 0 for all dim
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return _unimplemented(name, 'output size that are not factor of input size')
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
# Arguments:
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


@parse_args('v', 'is', 'f')
def constant_pad_nd(g, input, padding, value):
    mode = "constant"
    paddings = _prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode, value_f=value)


@parse_args('v', 'is')
def reflection_pad(g, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


@parse_args('v', 'is')
def replication_pad(g, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, align_corners=None):
        sym_help._interpolate_warning(interpolate_mode)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        if align_corners:
            return _unimplemented(name, "align_corners == True")
        scales = sym_help._interpolate_size_to_scales(g, input, output_size, dim)
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)
    return symbolic_fn


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")


def __interpolate(g, input, size, scale_factor, mode , align_corners):
    scales, mode = sym_help._interpolate_get_scales_and_mode(g, input, size, scale_factor,
                                                             mode , align_corners)
    return g.op("Upsample", input, scales, mode_s=mode)


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
    return g.op("Greater", input, other)


def lt(g, input, other):
    return lt_impl(g, input, other)


def lt_impl(g, input, other):
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


def where(g, condition, self, other):
    return g.op("Where", condition, self, other)


@parse_args('v', 'i', 'none')
def log_softmax(g, input, dim, dtype=None):
    # PyTorch dim and ONNX axis have different meanings.
    # See Softmax comment for details.
    # TODO: remove this as onnx opset 11 spec allows negative axes
    if dim < 0:
        dim = input.type().dim() + dim
    if input.type().dim() != dim + 1:
        return _unimplemented("dim", "ONNX and PyTorch use different strategies to split the input.")
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != 'prim::Constant':
        return_op = g.op("Cast", return_op, to_i=sym_help.scalar_type_to_onnx[dtype])
    return return_op


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is', 'i', 'i', 'i', 'i')
def _convolution(g, input, weight, bias, stride, padding, dilation,
                 transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled):
    weight_size = weight.type().sizes()

    args = [input, weight]
    # ONNX only supports 1D bias
    if not sym_help._is_none(bias) and bias.type().dim() == 1:
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

    if not sym_help._is_none(bias) and bias.type().dim() != 1:
        return g.op("Add", n, bias)
    else:
        return n


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    input_sizes = input.type().sizes()
    if len(input_sizes) == 2:
        # batchnorm1d accepts 2d and 3d array, but ONNX only accepts 3d
        input = g.op("Unsqueeze", input, axes_i=[2])

    if weight is None or sym_help._is_none(weight):
        assert len(input_sizes) > 1
        weight_value = torch.tensor([1.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        assert len(input_sizes) > 1
        bias_value = torch.tensor([0.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               outputs=1 if not training else 5)
    if not training:
        if len(input_sizes) == 2:
            out = g.op("Squeeze", out, axes_i=[2])
        return out
    else:
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        if len(input_sizes) == 2:
            res = g.op("Squeeze", res, axes_i=[2])
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

    layer_norm = div(g, numerator, denominator)

    if not (weight is None or sym_help._is_none(weight)):
        layer_norm = mul(g, layer_norm, weight)
    if not (bias is None or sym_help._is_none(bias)):
        layer_norm = add(g, layer_norm, bias)

    return layer_norm


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def instance_norm(g, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled):
    input_sizes = input.type().sizes()
    if weight is None or sym_help._is_none(weight):
        assert len(input_sizes) > 1
        weight_value = torch.tensor([1.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        assert len(input_sizes) > 1
        bias_value = torch.tensor([0.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)
    return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)


@parse_args('v', 'i', 'i', 'i')
def unfold(g, input, dimension, size, step):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)
    if input.isCompleteTensor():
        sizedim = input.type().sizes()[dimension]
        low_indices = range(0, sizedim, step)
        hi_indices = range(size, sizedim + 1, step)
        stack = [sym_help._slice_helper(g, input, axes=[dimension], starts=[low], ends=[hi])
                 for low, hi in zip(low_indices, hi_indices)]
        ndim = input.type().dim()
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))
        unsqueeze = [g.op("Unsqueeze", g.op("Transpose", t, perm_i=perm), axes_i=[dimension]) for t in stack]
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

    index_const = sym_help._maybe_get_scalar(index)
    index_dim = index.type().dim()
    if not sym_help._is_value(index_const):
        # Index is a constant scalar. Make it a size 1 constant tensor.
        index = g.op("Constant", value_t=torch.LongTensor([index_const]))
    elif index_dim is not None:
        if index_dim == 0:
            # Index is a scalar. Reshape it to a size 1 tensor.
            index = g.op("Reshape", index, g.op("Constant", value_t=torch.LongTensor([1])))
    return g.op("Gather", self, index, axis_i=dim)


def index_put(g, self, indices_list_value, values, accumulate):
    indices_list = sym_help._unpack_list(indices_list_value)
    args = [self] + indices_list + [values, accumulate]
    return g.op("ATen", *args, operator_s='index_put')


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
    if self.isCompleteTensor() and other.isCompleteTensor() and self.type().scalarType() == other.type().scalarType():
        return self

    if other.isCompleteTensor():
        other_type_name = other.type().scalarType()
        return g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[other_type_name])
    else:
        # We don't know the type of other, bail by emitting ATen
        return g.op("ATen", self, other, operator_s="type_as")


@parse_args('v', 'v', 'i', 'f')
def cosine_similarity(g, x1, x2, dim, eps):
    return g.op("ATen", x1, x2, dim_i=dim, eps_f=eps, operator_s="cosine_similarity")


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
    return g.op("Pow", self, exponent)


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
    if not train:  # in eval mode, dropout is non-op
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
    return g.op("ATen", input, weight, bias, operator_s="conv_tbc", pad_i=pad)


@parse_args('v', 'i', 'i')
def _unique(g, input, sorted, return_inverse):
    return g.op("ATen", input, operator_s="_unique", sorted_i=sorted,
                return_inverse_i=return_inverse, outputs=2)


@parse_args('v', 'i', 'i', 'i')
def _unique2(g, input, sorted, return_inverse, return_counts):
    return g.op("ATen", input, operator_s="_unique2", sorted_i=sorted,
                return_inverse_i=return_inverse, return_counts_i=return_counts,
                outputs=3)


for k, v in sym_help.cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = parse_args('v', 'i')(partial(sym_help._cast_func_template, v))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


def scalar_tensor(g, scalar, dtype, *options):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    scalar = g.op("Cast", scalar, to_i=sym_help.scalar_type_to_onnx[dtype])
    return scalar


@parse_args('v', 'i', 'v', 'v', 'v')
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device, layout and pin_memory in ONNX, so we ignore it
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def zeros_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([0], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v', 'v')
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([1], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def ones_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([1], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    if dtype is None:
        dtype = 6  # float
    const_value = sym_help._maybe_get_const(value, 't')
    if sym_help._is_value(const_value):
        tmp = zeros(g, sizes, dtype, layout, device)
        return add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = sym_help._get_const(dtype, 'i', 'dtype')
        return g.op("ConstantOfShape", sizes,
                    value_t=torch.tensor([const_value], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'f', 'i', 'v', 'v', 'v', 'v')
def full_like(g, input, fill_value, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    if dtype is None:
        dtype = 6  # float
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([fill_value], dtype=sym_help.scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, self, dim, start, end, step):
    if step != 1:
        _unimplemented("slice", "step!=1 is currently not supported")
    if start.node().kind() != 'onnx::Constant' or \
            end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant':
        if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX:
            raise RuntimeError('Unsupported: ONNX export of Slice with dynamic inputs. DynamicSlice '
                               'is a deprecated experimental op. Please use statically allocated '
                               'variables or export to a higher opset version.')
        else:
            start_unsqueezed = g.op("Unsqueeze", start, axes_i=[0])
            end_unsqueezed = g.op("Unsqueeze", end, axes_i=[0])
            dim_unsqueezed = g.op("Unsqueeze", dim, axes_i=[0])
            return g.op("DynamicSlice", self, start_unsqueezed, end_unsqueezed, dim_unsqueezed)
    else:
        start = _parse_arg(start, 'i')
        end = _parse_arg(end, 'i')
        dim = _parse_arg(dim, 'i')
        return sym_help._slice_helper(g, self, axes=[dim], starts=[start], ends=[end])


@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    return g.op("Clip", self, min_f=min_val, max_f=max_val)


def alias(g, self):
    return self


@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    # Handle negative dim
    if dim < 0:
        rank = self.type().dim()
        if rank:
            warnings.warn("ONNX export unsqueeze with negative axis " + str(dim) +
                          " might cause the onnx model to be incorrect. " +
                          "Negative axis is not supported in ONNX. " +
                          "Axis is converted to " + str(dim + rank + 1) +
                          " based on input shape at export time. " +
                          "Passing an tensor of different rank in execution will be incorrect.")
            dim = dim + rank + 1
        else:
            return _unimplemented('unsqueeze', 'negative axis with unknown input rank')

    return g.op("Unsqueeze", self, axes_i=[dim])

@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported for sort")
    if not self.isCompleteTensor():
        return _unimplemented("Sort", "input size not accessible")

    return g.op("TopK", self, k_i=self.type().sizes()[dim], axis_i=dim, outputs=2)

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
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            dtype = sym_help._get_const(args[0], 'i', 'dtype')
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
    if not sym_help._is_value(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    const_repeats = sym_help._maybe_get_const(repeats, 'is')

    if self.isCompleteTensor() and not sym_help._is_value(const_repeats):
        sizes = self.type().sizes()
        diff_dims = len(const_repeats) - len(sizes)
        if diff_dims > 0:
            self = view(g, self, [1] * diff_dims + sizes)
    return g.op("Tile", self, repeats)


@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    dims = self.type().sizes()
    if len(dims) != 4:
        return _unimplemented("pixel_shuffle", "only support 4d input")
    output_channel = dims[1] // upscale_factor // upscale_factor
    after_view = view(g, self, [-1, output_channel, upscale_factor, upscale_factor,
                                dims[2], dims[3]])
    after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
    return view(g, after_transpose,
                [-1, output_channel, dims[2] * upscale_factor, dims[3] *
                 upscale_factor])


def _generic_rnn(g, variant, input, initial_states, all_weights, has_biases,
                 num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None):

    warnings.warn("Exporting a model to ONNX with a batch_size other than 1, " +
                  "with a variable lenght with " + variant + " can cause an error " +
                  "when running the ONNX model with a different batch size. " +
                  "Make sure to save the model with a batch size of 1, " +
                  "or define the initial states (h0/c0) as inputs of the model. ")

    onnxActivations = ['Relu', 'Tanh', 'Sigmoid', 'Affine', 'LeakyRelu', 'ThresholdedRelu',
                       'ScaledTanh', 'HardSigmoid', 'Elu', 'Softsign', 'Softplus']
    variantToOnnxActivationMap = dict(zip([act_fun.lower() for act_fun in onnxActivations], onnxActivations))
    weights_per_layer = 4 if has_biases else 2
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
    hidden_size = w_hh.type().sizes()[1]

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

    def transform_weights(layer_index):
        if variant == 'RNN':
            weight_ih, weight_hh, bias_ih, bias_hh = layer_weights[layer_index]
        elif variant == 'GRU' or variant == 'LSTM':
            weight_ih, weight_hh, bias_ih, bias_hh = \
                [reform_weights(g, w, hidden_size, reform_permutation) for w in layer_weights[layer_index]]
        bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

        return tuple(g.op('Unsqueeze', x, axes_i=[0]) for x in (weight_ih, weight_hh, bias_concat))

    def retrieve_state(x, start, end):
        return x if num_layers == 1 else sym_help._slice_helper(g, x, axes=[0], starts=[start], ends=[end])

    for i in range(num_layers):
        if unidirectional:
            weight_ih, weight_hh, bias_concat = transform_weights(i)
            state_indices = i, i + 1
        else:
            weight_ih_f, weight_hh_f, bias_f = transform_weights(2 * i)
            weight_ih_b, weight_hh_b, bias_b = transform_weights(2 * i + 1)

            weight_ih = g.op('Concat', weight_ih_f, weight_ih_b, axis_i=0)
            weight_hh = g.op('Concat', weight_hh_f, weight_hh_b, axis_i=0)
            bias_concat = g.op('Concat', bias_f, bias_b, axis_i=0)

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
            prev_output = g.op('Squeeze', prev_output, axes_i=[1])

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
        lengths = _cast_Int(g, lengths, False)
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


def randn(g, *shapes):
    shapes_list = list(shapes)
    shape = sym_help._get_const(shapes_list[0], "is", "randn")
    return g.op('RandomNormal', shape_i=shape)


def rand(g, *shapes):
    shapes_list = list(shapes)
    shape = sym_help._get_const(shapes_list[0], "is", "rand")
    return g.op('RandomUniform', shape_i=shape)


def randn_like(g, self, dtype, layout, device, pin_memory=False, memory_format=None):
    dtype = sym_help._get_const(dtype, 'i', 'dtype')
    if dtype is None:
        dtype = 6  # float
    return g.op('RandomNormalLike', self, dtype_i=sym_help.scalar_type_to_onnx[dtype])


def rand_like(g, self, dtype, layout, device, pin_memory=False, memory_format=None):
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
    dim = input.type().dim()
    # TODO: remove this as onnx opset 11 spec allows negative axes
    if end_dim < 0 :
        end_dim = dim + end_dim
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1 and end_dim == dim - 1 :
        return g.op("Flatten", input, axis_i=start_dim)
    if start_dim == 0 and end_dim == dim - 2 :
        return g.op("Flatten", input, axis_i=end_dim + 1)
    # use Reshape for cases where the output shape is not 2D
    if not input.isCompleteTensor():
        return _unimplemented("flatten", "input size not accessible")
    input_dims = input.type().sizes()
    output_dims = []
    for i in range(0, dim):
        if start_dim < i and end_dim >= i:
            output_dims[start_dim] = output_dims[start_dim] * input_dims[i]
        else:
            output_dims.append(input_dims[i])
    shape = g.op("Constant", value_t=torch.LongTensor(output_dims))
    p = _reshape_from_tensor(g, input, shape)
    return p


@parse_args('v')
def nonzero(g, input):
    return t(g, g.op('NonZero', input))


@parse_args('v')
def isnan(g, input):
    output = g.op('IsNaN', input)
    return output


@parse_args('v', 'i', 'i', 'i')
def narrow(g, input, dim, start, length):
    return sym_help._slice_helper(g, input, axes=[dim], starts=[start], ends=[start + length])


def argmax(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        flattened = reshape(g, input, (-1,))
        return g.op('ArgMax', flattened, axis_i=0, keepdims_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMax', input, axis_i=dim, keepdims_i=keepdim)


def argmin(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        flattened = reshape(g, input, (-1,))
        return g.op('ArgMin', flattened, axis_i=0, keepdims_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMin', input, axis_i=dim, keepdims_i=keepdim)


@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    return g.op("Scatter", self, index, src, axis_i=dim)


@parse_args('v', 'i', 'v', 'v')
def scatter_add(g, self, dim, index, src):
    if not self.isCompleteTensor():
        return _unimplemented("scatter_add", "input size not accessible")
    dtype = self.type().scalarType()
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    dtype = sym_help.scalar_type_to_pytorch_type[dtype]
    sizes = self.type().sizes()
    to_add = g.op("Constant", value_t=torch.zeros(sizes, dtype=dtype))
    to_add = sym_help._scatter_helper(g, to_add, dim, index, src)
    return add(g, self, to_add)


def log2(g, self):
    _ln2 = 0.693147180559945309
    return g.op('Div', log(g, self), g.op('Constant', value_t=torch.Tensor([_ln2])))


def prim_shape(g, self):
    return g.op('Shape', self)


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
    mul = g.op("Mul", g.op("Unsqueeze", self, axes_i=[dim + 1]), index)
    return g.op("ReduceSum", mul, axes_i=[dim], keepdims_i=0)


@parse_args('v', 'is', 'b', 'i')
def _std(g, input, dim, unbiased, keepdim):
    if input.isCompleteTensor():
        sqrd = g.op("Mul", input, input)
        if dim is None:
            sqrdmean = g.op("ReduceMean", sqrd, keepdims_i=0)
            mean = g.op("ReduceMean", input, keepdims_i=0)
            redudced_dims = input.type().sizes()
        else:
            sqrdmean = g.op("ReduceMean", sqrd, axes_i=dim, keepdims_i=keepdim)
            mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
            redudced_dims = [input.type().sizes()[i] for i in dim]
        meansqrd = g.op("Mul", mean, mean)
        var = g.op("Abs", g.op("Sub", sqrdmean, meansqrd))
        # This is to correct bias in calculating variance, by dividing it over (N - 1) instead on N
        if unbiased:
            count = numpy.prod(redudced_dims)
            mul = g.op("Mul", var, g.op("Constant", value_t=torch.tensor(count, dtype=torch.float)))
            var = g.op("Div", mul, g.op("Constant", value_t=torch.tensor(count - 1, dtype=torch.float)))
        std = g.op("Sqrt", var)
        return std
    else:
        _unimplemented("std", "Unknown input rank. Cannot compute std along dimensions.")


# Since position of optional arguments can change for std, this is a hack to find if first argument
# is 'dim' or 'unbiased'. As shown below, 'dim' argument could be listed before 'unbiased' :
# torch.std(input, unbiased=True)
# torch.std(input, dim, keepdim=False, unbiased=True)
def std(g, input, *args):
    if args[0].type().isSubtypeOf(ListType.ofInts()):
        return _std(g, input, *args)
    else:
        return _std(g, input, None, args[0], None)


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

    if len(args) == 5:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[1])
        end = g.op("Unsqueeze", args[0], axes_i=[0])
        arange_tensor = g.op("Squeeze", nonzero(g, ones(g, end, dtype, *(args[2:]))), axes_i=[1])
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        end = g.op("Unsqueeze", args[1], axes_i=[0])
        start = g.op("Unsqueeze", args[0], axes_i=[0])
        range_tensor = g.op("Sub", end, start)
        arange_tensor = g.op("Add", g.op("Squeeze", nonzero(g, ones(g, range_tensor, dtype, *(args[3:]))), axes_i=[1]), start)
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[3])
        step = g.op("Unsqueeze", args[2], axes_i=[0])
        end = g.op("Unsqueeze", args[1], axes_i=[0])
        start = g.op("Unsqueeze", args[0], axes_i=[0])
        range_tensor = g.op("Div", g.op("Sub", end, start), step)
        arange_tensor = g.op("Squeeze", nonzero(g, ones(g, range_tensor, dtype, *(args[4:]))), axes_i=[1])
        arange_tensor = g.op("Add", g.op("Mul", arange_tensor, step), start)
        return g.op("Cast", arange_tensor, to_i=sym_help.scalar_type_to_onnx[dtype])
    else:
        raise NotImplementedError("Unknown aten::arange signature taking " + str(len(args)) + " arguments.")


def masked_fill(g, self, mask, value):
    mask = _cast_Bool(g, mask, False)
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
        if not sym_help._is_none(index) and index.type().scalarType() == "Byte":
            warnings.warn("Exporting aten::index operator with indices of type Byte. "
                          "Only 1-D indices are supported. In any other case, "
                          "this will produce an incorrect ONNX graph.")
            index = squeeze(g, nonzero(g, index), dim=1)
        return index

    indices = [try_mask_to_index(idx) for idx in indices]
    if len(indices) == 1:
        return index_select(g, self, 0, indices[0])
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
            rank = self.type().dim()
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
            rank = self.type().dim()
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
    sumsqr = g.op('ReduceSum', sqr, axes_i=dim, keepdims_i=keepdim)
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
    tensors = [view(g, t, torch.LongTensor([-1])) for t in sym_help._unpack_list(tensor_list)]
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
    if sym_help._is_fp(input):
        div = g.op("Floor", div)
    quo = g.op("Mul", div, other)
    return g.op("Sub", input, quo)

def gelu(g, self):
    _sqrt2 = 1.4142135623730951
    erf = g.op('Erf', div(g, self, torch.tensor(_sqrt2)))
    erf_plusone = add(g, erf, g.op('Constant', value_t=torch.tensor(1, dtype=torch.float)))
    return mul(g, mul(g, self, erf_plusone), g.op('Constant', value_t=torch.tensor(0.5, dtype=torch.float)))


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, weight, bias, num_groups_i=num_groups,
                    eps_f=eps, cudnn_enabled_i=cudnn_enabled, operator_s="group_norm")

    input_sizes = input.type().sizes()
    assert input_sizes[1] % num_groups == 0
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
    axes = [i for i in range(1, len(input_sizes) - 1)]
    return add(g, mul(g, norm, g.op("Unsqueeze", weight, axes_i=axes)), g.op("Unsqueeze", bias, axes_i=axes))

@parse_args('v', 'v', 'i')
def _weight_norm(g, weight_v, weight_g, dim):
    rank = weight_v.type().dim()
    if rank:
        # W = g * ((v) / ||v||)
        # Compute norm_except_dim for l2 norm. dim = None means over all dims
        # torch's weight_norm module sets dim = -1 if it's None.
        # This conflicts the logic for negative axes to access dims backwards
        # TODO: Might need a fix in torch group_norm module
        axes = list(range(rank))
        if dim:
            if dim < -1:
                dim += rank
            if dim != -1:
                axes.remove(dim)
        norm_v = norm(g, weight_v, 2, axes, 1)
        div = g.op("Div", weight_v, norm_v)
        return g.op("Mul", div, weight_g)
    else:
        return g.op("ATen", weight_v, weight_g, dim_i=dim, operator_s="_weight_norm")

# Ops below are for PyTorch Quantization conversion process.
@parse_args('v', 'f', 'i', 't')
def quantize_per_tensor(g, input, scale, zero_point, dtype):
    return sym_caffe2.quantize_per_tensor(g, input, scale, zero_point)

@parse_args('v')
def dequantize(g, input):
    return sym_caffe2.dequantize(g, input)

@parse_args('v', 't', 't', 't', 't', 't', 't', 't')
def _empty_affine_quantized(g, input, shape, scale, zero_point, dtype, pin_memory, memory_format, layout):
    return input
