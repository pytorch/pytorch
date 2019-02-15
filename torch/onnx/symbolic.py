import numbers

import torch
from torch._C import DynamicType, ListType
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils.rnn import PackedSequence
import warnings

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from collections import Iterable
from functools import partial, wraps
import itertools

# EDITING THIS FILE? READ THIS FIRST!
#
# - This file is ONLY for ATen operators (e.g., operators that show up in the
#   trace as aten::blah).  If you need to special case a primitive operator,
#   look at _run_symbolic_function
# - Parameter ordering does NOT necessarily match what is in VariableType.cpp;
#   tensors are always first, then non-tensor arguments.
# - Parameter names must *exactly* match the names in VariableType.cpp, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by the trailing underscore, and
#   transparently dispatched to their non inplace versions in
#   'run_symbolic_function'.   See Note [Export inplace]
#
# ---------------------------------------------------------------------
# A note on Tensor types
# ---------------------------------------------------------------------
#
# In general, we should avoid depending on the type of Tensor Values contained
# within the trace graph. However, this is sometimes unavoidable (due to ONNX
# spec requirements, etc). If you are implementing a symbolic and need Tensor
# type information, note that there are several levels of Tensor types, defined
# in aten/src/ATen/core/jit_type.h:
#
# DynamicType - This is a Tensor, but we don't know anything about its
#               properties (e.g. scalar type, # dims, shapes).
#               Appears as `Tensor` in graph print-outs.
# UndefinedTensorType <: DynamicType - Denotes an undefined Tensor
# TensorType <: DynamicType - Denotes a Tensor for which we know the scalar
#                             type and number of dimensions, but not the concrete
#                             shapes. For example, appears as 'Float(*, *)' in
#                             graph print-outs. Useful accessor methods include
#                             dim() and scalarType()
# CompleteTensorType <: TensorType - Denotes a Tensor for which we know the
#                                    concrete sizes in addition to the information
#                                    contained in TensorTyper. This adds a sizes()
#                                    method which can be used to retrieve the
#                                    concrete sizes.
#
# In general, we should prefer to rely on the least specific information possible.
# For example, not relying on tensor properties at all is better than relying
# on the number of dimensions (TensorType) which is better than relying on
# concrete shapes (CompleteTensorType). Doing so will make the export symbolics
# more robust to different graphs.

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

# Save some builtins as locals, because we'll shadown them below
_sum = sum


def _parse_arg(value, desc):
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    if value.node().kind() != 'onnx::Constant':
        raise RuntimeError("ONNX symbolic expected a constant value in the trace")
    tval = value.node()['value']
    if desc == 'i':
        return int(tval)
    elif desc == 'f':
        return float(tval)
    elif desc == 't':
        return tval
    elif desc == 'is':
        return [int(v) for v in tval]
    else:
        raise RuntimeError("Casting constants to `{}` is not implemented".format(desc))


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == 'onnx::Constant':
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if _is_value(value) and value.node().kind() != 'onnx::Constant':
        raise RuntimeError("ONNX symbolic expected a constant value of the {} argument".format(arg_name))
    return _parse_arg(value, desc)


def _unpack_list(list_value):
    list_node = list_value.node()
    assert list_node.kind() == "prim::ListConstruct"
    return list(list_node.inputs())


def parse_args(*arg_descriptors):
    def decorator(fn):
        def wrapper(g, *args):
            assert len(arg_descriptors) == len(args)
            args = [_parse_arg(arg, arg_desc) for arg, arg_desc in zip(args, arg_descriptors)]
            return fn(g, *args)
        # In Python 2 functools.wraps chokes on partially applied functions, so we need this as a workaround
        try:
            wrapper = wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper
    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self
    elif tensor.type().kind() == "DimensionedTensorType" or tensor.type().kind() == "CompleteTensorType":
        ty = tensor.type().scalarType().lower()
        return getattr(self, ty)()
    else:
        return self


def _is_value(x):
    return isinstance(x, torch._C.Value)


def _is_tensor_list(x):
    return x.type().isSubtypeOf(ListType.ofTensors())


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def _try_get_scalar_type(*args):
    for arg in args:
        try:
            return arg.type().scalarType()
        except RuntimeError:
            pass
    return None


# ---------------------------------------------------------------------
# ONNX operator version
# ---------------------------------------------------------------------

# READ ME BEFORE EDITING _onnx_opset_version:
#
# The variable below controls which ONNX operator set version we are
# targeting.   THIS VARIABLE HAS SEMANTIC EFFECT!  Say a breaking
# change occurred in version 8.  As long as this variable < 8, you can
# export models targeting the old behavior.  However, if you bump
# this variable to 8 or later, the breaking change will take into effect:
# you MUST adjust any symbolic affected by breaking changes.  The ONNX
# spec publishes a *comprehensive* list of BC-breaking changes for every
# operator revision at:
#
#   https://github.com/onnx/onnx/blob/master/docs/Changelog.md
#
# Please be sure to go through and check all of our implementations here before
# increasing this number.  This includes symbolic definitions NOT in this
# file, so grep for "OpName" (with quotes)

_onnx_opset_version = 9


# ---------------------------------------------------------------------
# Symbolic definitions
# ---------------------------------------------------------------------


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
    return g.op("prim::None")


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
    if alpha and _scalar(_maybe_get_scalar(alpha)) != 1:
        return _unimplemented("add", "alpha != 1")
    # See Note [Pointwise by scalar]
    other = _maybe_get_scalar(other)
    return g.op("Add", self, _if_scalar_type_as(g, other, self))


def sub(g, self, other, alpha=None):
    # default alpha arg is to allow no-alpha sub (aten sub st overload no alpha)
    if alpha and _scalar(_maybe_get_scalar(alpha)) != 1:
        return _unimplemented("sub", "alpha != 1")
    # See Note [Pointwise by scalar]. Note that self or other may be scalars.
    other = _maybe_get_scalar(other)
    return g.op("Sub", self, _if_scalar_type_as(g, other, self))


def rsub(g, self, other, alpha=None):
    other = _maybe_get_scalar(other)
    other = _if_scalar_type_as(g, other, self)
    return sub(g, other, self, alpha=alpha)


def mul(g, self, other):
    # See Note [Pointwise by scalar]
    other = _maybe_get_scalar(other)
    return g.op("Mul", self, _if_scalar_type_as(g, other, self))


def div(g, self, other):
    # See Note [Pointwise by scalar]
    other = _maybe_get_scalar(other)
    return g.op("Div", self, _if_scalar_type_as(g, other, self))


def reciprocal(g, self):
    return g.op("Div", _if_scalar_type_as(g, torch.ones(1), self), self)


@parse_args('v', 'i')
def cat(g, tensor_list, dim):
    tensors = _unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@parse_args('v', 'i')
def stack(g, tensor_list, dim):
    unsqueezed = [g.op("Unsqueeze", t, axes_i=[dim]) for t in _unpack_list(tensor_list)]
    return g.op("Concat", *unsqueezed, axis_i=dim)


def mm(g, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    ty = _try_get_scalar_type(self, other).lower()
    C = g.constant(0, [1], ty)
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


def bmm(g, self, other):
    return g.op("MatMul", self, other)


def matmul(g, self, other):
    return g.op("MatMul", self, other)


@parse_args('v', 'v', 'v', 't', 't')
def addmm(g, self, mat1, mat2, beta, alpha):
    return g.op("Gemm", mat1, mat2, self, beta_f=_scalar(beta), alpha_f=_scalar(alpha))


def neg(g, self):
    return g.op("Neg", self)


def sqrt(g, self):
    return g.op("Sqrt", self)


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


def _reduce_op_symbolic(onnx_op_name):
    def symbolic(g, self, dim=None, keepdim=None):
        if dim is None:
            # all-reduce path
            return g.op(onnx_op_name, self, keepdims_i=0)
        else:
            # dim-reduce path
            dim, keepdim = _get_const(dim, 'i', 'dim'), _get_const(keepdim, 'i', 'keepdim')
            return g.op(onnx_op_name, self, axes_i=[dim], keepdims_i=keepdim)
    return symbolic

mean = _reduce_op_symbolic('ReduceMean')
sum = _reduce_op_symbolic('ReduceSum')
prod = _reduce_op_symbolic('ReduceProd')


@parse_args('v', 'i')
def cumsum(g, input, dim):
    return g.op("ATen", input, operator_s="cumsum", dim_i=dim)


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def expand(g, self, size, implicit):
    size = _maybe_get_const(size, 'is')
    if not _is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Expand", self, size)


def expand_as(g, self, other):
    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op("Gather", weight, indices)


@parse_args('v', 'v', 'v', 'i', 'i', 'i')
def embedding_bag(g,
                  embedding_matrix,
                  indices,
                  offsets,
                  scale_grad_by_freq,
                  mode,
                  sparse):
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
    full_shape = g.op("Shape", self)
    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


@parse_args('v', 'i', 'i')
def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    axes = list(range(self.type().dim()))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return g.op("Transpose", self, perm_i=axes)


@parse_args('v', 'is')
def permute(g, self, dims):
    if dims == list(range(0, len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


def view(g, self, size):
    size = _maybe_get_const(size, 'is')
    if _is_value(size):
        shape = size
    else:
        if self.isTensor():
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


@parse_args('v', 'i', 'v')
def select(g, self, dim, index):
    if dim > 1:
        # TODO: this is a temporary hack because of the implementation details
        # of Gather in caffe2. We need to change this as soon as possible.
        # TODO: this breaks if index == -1
        index_val = _parse_arg(index, 'i')
        slice_node = g.op("Slice", self, axes_i=[dim], starts_i=[index_val], ends_i=[index_val + 1])
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
        dims = [_get_const(dim, 'i', 'dim')]
    return g.op("Squeeze", self, axes_i=dims)


def prelu(g, self, weight):
    return g.op("PRelu", self, weight)


def relu(g, input):
    return g.op("Relu", input)


@parse_args('v', 't', 't')
def threshold(g, self, threshold, value):
    # See Note [Export inplace]
    if _scalar(threshold) != 0:
        return _unimplemented("threshold", "non-zero threshold")
    if _scalar(value) != 0:
        return _unimplemented("threshold", "non-zero value")
    return g.op("Relu", self)


def leaky_relu(g, input, negative_slope, inplace=False):
    negative_slope = _get_const(negative_slope, 't', 'negative_slope')
    # See Note [Export inplace]
    # TODO: Talk to ONNX about unconditional cast of scalar to float
    return g.op("LeakyRelu", input, alpha_f=_scalar(negative_slope))


@parse_args('v', 'i')
def glu(g, input, dim):
    assert input.type().sizes()[dim] % 2 == 0

    first, second = g.op('Split', input, axis_i=dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))


@parse_args('v', 'i')
def softmax(g, input, dim):
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
    if dim < 0:
        dim = input.type().dim() + dim
    if input.type().dim() != dim + 1:
        return _unimplemented("dim", "ONNX and PyTorch use different strategies to split the input.")
    return g.op('Softmax', input, axis_i=dim)


@parse_args('v', 't', 'v')
def softplus(g, self, beta, threshold):
    if beta != 1:
        return _unimplemented("beta", "has to be 1")
    return g.op('Softplus', self)


@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool1d_with_indices(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool1d_with_indices", "ceil_mode")
    if set(_single(dilation)) != {1}:
        return _unimplemented("max_pool1d_with_indices", "dilation")
    if stride is None:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_single(kernel_size),
             pads_i=_single(padding) * 2,
             strides_i=_single(stride))
    return r, None


@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool2d_with_indices(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool2d_with_indices", "ceil_mode")
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d_with_indices", "dilation")
    if not stride:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_pair(kernel_size),
             pads_i=_pair(padding) * 2,
             strides_i=_pair(stride))
    return r, None


@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool3d_with_indices(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool3d_with_indices", "ceil_mode")
    if set(_triple(dilation)) != {1}:
        return _unimplemented("max_pool3d_with_indices", "dilation")
    if not stride:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_triple(kernel_size),
             pads_i=_triple(padding) * 2,
             strides_i=_triple(stride))
    return r, None


def _avg_pool(name, tuple_fn):
    @parse_args('v', 'is', 'is', 'is', 'i', 'i')
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
        if ceil_mode:
            return _unimplemented("avg_pool2d", "ceil_mode")
        if not stride:
            stride = kernel_size

        padding = tuple(tuple_fn(padding))
        if count_include_pad:
            input = g.op("Pad", input,
                         pads_i=((0,) * 2 + padding) * 2,
                         mode_s='constant',
                         value_f=0.)
            padding = (0,) * len(padding)

        return g.op("AveragePool", input,
                    kernel_shape_i=tuple_fn(kernel_size),
                    strides_i=tuple_fn(stride),
                    pads_i=padding * 2)
    return symbolic_fn


avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)


@parse_args('v', 'is')
def adaptive_avg_pool2d(g, input, output_size):
    assert output_size == [1, 1], "Only output_size=[1, 1] is supported"
    return g.op("GlobalAveragePool", input)


@parse_args('v', 'is')
def adaptive_max_pool2d(g, input, output_size):
    assert output_size == [1, 1], "Only output_size=[1, 1] is supported"
    return g.op("GlobalMaxPool", input), None


@parse_args('v', 'is', 'f')
def constant_pad_nd(g, input, padding, value):
    from torch.autograd._functions.utils import prepare_onnx_paddings
    mode = "constant"
    paddings = prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode, value_f=value)


@parse_args('v', 'is')
def reflection_pad(g, input, padding):
    from torch.autograd._functions.utils import prepare_onnx_paddings
    mode = "reflect"
    paddings = prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


@parse_args('v', 'is')
def replication_pad(g, input, padding):
    from torch.autograd._functions.utils import prepare_onnx_paddings
    mode = "edge"
    paddings = prepare_onnx_paddings(input.type().dim(), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


@parse_args('v', 'is')
def upsample_nearest2d(g, input, output_size):
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]
    scales = g.op("Constant", value_t=torch.tensor([1., 1., height_scale,
                                                    width_scale]))

    return g.op("Upsample", input, scales,
                mode_s="nearest")


@parse_args('v', 'is', 'i')
def upsample_bilinear2d(g, input, output_size, align_corners):
    if align_corners:
        return _unimplemented("upsample_bilinear2d", "align_corners == True")
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]
    scales = g.op("Constant", value_t=torch.tensor([1., 1., height_scale,
                                                    width_scale]))
    return g.op("Upsample", input, scales,
                mode_s="linear")


def wrap_logical_op_with_cast_to_uint8(func):
    def wrap_with_cast(g, input, other):
        return g.op("Cast", func(g, input, other), to_i=cast_pytorch_to_onnx['Byte'])
    return wrap_with_cast


def wrap_logical_op_with_negation(func):
    def wrap_with_not(g, input, other):
        return g.op("Not", func(g, input, other))
    return wrap_with_not


@wrap_logical_op_with_cast_to_uint8
def eq(g, self, other):
    return g.op("Equal", self, other)


@wrap_logical_op_with_cast_to_uint8
@wrap_logical_op_with_negation
def ne(g, self, other):
    return g.op("Equal", self, other)


@wrap_logical_op_with_cast_to_uint8
def gt(g, input, other):
    return gt_impl(g, input, other)


def gt_impl(g, input, other):
    other = _maybe_get_scalar(other)
    return g.op("Greater", input, _if_scalar_type_as(g, other, input))


@wrap_logical_op_with_cast_to_uint8
def lt(g, input, other):
    return lt_impl(g, input, other)


def lt_impl(g, input, other):
    other = _maybe_get_scalar(other)
    return g.op("Less", input, _if_scalar_type_as(g, other, input))


@wrap_logical_op_with_cast_to_uint8
@wrap_logical_op_with_negation
def ge(g, input, other):
    other = _maybe_get_scalar(other)
    return lt_impl(g, input, _if_scalar_type_as(g, other, input))


@wrap_logical_op_with_cast_to_uint8
@wrap_logical_op_with_negation
def le(g, input, other):
    other = _maybe_get_scalar(other)
    return gt_impl(g, input, _if_scalar_type_as(g, other, input))


def where(g, condition, self, other):
    return g.op("ATen", condition, self, other, operator_s="where")


@parse_args('v', 'i')
def log_softmax(g, input, dim=None):
    # PyTorch dim and ONNX axis have different meanings.
    # See Softmax comment for details.
    if dim < 0:
        dim = input.type().dim() + dim
    if input.type().dim() != dim + 1:
        return _unimplemented("dim", "ONNX and PyTorch use different strategies to split the input.")
    return g.op("LogSoftmax", input, axis_i=dim)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is', 'i', 'i', 'i', 'i')
def _convolution(g, input, weight, bias, stride, padding, dilation,
                 transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled):
    weight_size = weight.type().sizes()

    args = [input, weight]
    # ONNX only supports 1D bias
    if bias.node().kind() != "prim::None" and bias.type().dim() == 1:
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

    if bias.node().kind() != "prim::None" and bias.type().dim() != 1:
        return g.op("Add", n, bias)
    else:
        return n


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    input_sizes = input.type().sizes()
    if len(input_sizes) == 2:
        # batchnorm1d accepts 2d and 3d array, but ONNX only accepts 3d
        input = g.op("Unsqueeze", input, axes_i=[2])

    if weight is None or weight.node().kind() == "prim::None":
        assert len(input_sizes) > 1
        weight_value = torch.tensor([1.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().kind() == "prim::None":
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
        saved_mean.setUniqueName("batch_norm_dead_output-" + saved_mean.uniqueName())
        saved_var.setUniqueName("batch_norm_dead_output-" + saved_var.uniqueName())
        if len(input_sizes) == 2:
            res = g.op("Squeeze", res, axes_i=[2])
        return res


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def instance_norm(g, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled):
    input_sizes = input.type().sizes()
    if weight is None or weight.node().kind() == "prim::None":
        assert len(input_sizes) > 1
        weight_value = torch.tensor([1.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().kind() == "prim::None":
        assert len(input_sizes) > 1
        bias_value = torch.tensor([0.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)
    return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)


@parse_args('v', 'i', 'i', 'i')
def unfold(g, input, dimension, size, step):
    return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)


@parse_args('v', 'v', 'i')
def _weight_norm(graph, v, g, dim):
    return graph.op("ATen", v, g, dim_i=dim, operator_s="_weight_norm")


@parse_args('v', 't', 't', 't')
def elu(g, input, alpha, scale, input_scale):
    if scale and scale != 1.:
        return _unimplemented("scale", "does not support scale in Elu")
    if input_scale and input_scale != 1.:
        return _unimplemented("input_scale", "does not support input_scale in Elu")
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=_scalar(alpha))


def selu(g, input):
    return g.op("Selu", input)


@parse_args('v', 'i', 'v')
def index_select(g, self, dim, index):
    return g.op("Gather", self, index, axis_i=dim)


def index_put(g, self, indices_list_value, values, accumulate):
    indices_list = _unpack_list(indices_list_value)
    args = [self] + indices_list + [values, accumulate]
    return g.op("ATen", *args, operator_s='index_put')


def type_as(g, self, other):
    if self.isTensor() and other.isTensor() and self.type().scalarType() == other.type().scalarType():
        return self

    if other.isTensor():
        other_type_name = other.type().scalarType()
        return g.op("Cast", self, to_i=cast_pytorch_to_onnx[other_type_name])
    else:
        # We don't know the type of other, bail by emitting ATen
        return g.op("ATen", self, other, operator_s="type_as")


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, self, normalized_shape, weight, bias, eps, cudnn_enable):
    return g.op("ATen", self, weight, bias, normalized_shape_i=normalized_shape,
                eps_f=eps, cudnn_enable_i=cudnn_enable, operator_s="layer_norm")


# ignore clone operators that are inserted by PyTorch autograd
def clone(g, input):
    return input


def abs(g, self):
    return g.op("Abs", self)


def log(g, self):
    return g.op("Log", self)


def pow(g, self, exponent):
    exponent = _maybe_get_scalar(exponent)
    return g.op("Pow", self, _if_scalar_type_as(g, exponent, self))


def clamp(g, self, min, max):
    # min or max may be prim::None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax
    if min.node().kind() == "prim::None":
        return clamp_max(g, self, max)
    elif max.node().kind() == "prim::None":
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
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    if keepdim is None:
        return g.op("Max", self, dim_or_y)
    else:
        dim = _get_const(dim_or_y, 'i', 'dim')
        keepdim = _get_const(keepdim, 'i', 'keepdim')
        # TODO: export it as ReduceMax
        return g.op("ATen",
                    self,
                    operator_s="max",
                    dim_i=dim,
                    keepdim_i=keepdim,
                    outputs=2)


def min(g, self, dim_or_y=None, keepdim=None):
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    if keepdim is None:
        return g.op("Min", self, dim_or_y)
    else:
        dim = _get_const(dim_or_y, 'i', 'dim')
        keepdim = _get_const(keepdim, 'i', 'keepdim')
        # TODO: export it as ReduceMax
        return g.op("ATen",
                    self,
                    operator_s="min",
                    dim_i=dim,
                    keepdim_i=keepdim,
                    outputs=2)


def exp(g, self):
    return g.op("Exp", self)


@parse_args('v', 'f', 'i')
def dropout(g, input, p, train):
    if not train:  # in eval mode, dropout is non-op
        return input
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


def _unsupported_dropout(name):
    @parse_args('v', 'f', 'i')
    def feature_dropout(g, input, p, train):
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
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


@parse_args('v', 't', 'i', 'i')
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


# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute 'UINT8'
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
}

scalar_name_to_pytorch = {
    'uint8_t': 'Byte',
    'int8_t': 'Char',
    'double': 'Double',
    'float': 'Float',
    'half': 'Half',
    'int': 'Int',
    'int64_t': 'Long',
    'int16_t': 'Short',
}


# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/da7468853ae322252270bbb58032668bd21b7457/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,    # 0
    torch.int8,     # 1
    torch.short,    # 2
    torch.int,      # 3
    torch.int64,    # 4
    torch.half,     # 5
    torch.float,    # 6
    torch.double,   # 7
]


def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


for k, v in cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = parse_args('v', 'i')(partial(_cast_func_template, v))


scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
]


@parse_args('v', 'i', 'v', 'v')
def zeros(g, sizes, dtype, layout, device):
    # NOTE: no way to set device and layout in ONNX, so we ignore it
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor(0, dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def zeros_like(g, input, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor(0, dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def ones(g, sizes, dtype, layout, device):
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor(1, dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def ones_like(g, input, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor(1, dtype=scalar_type_to_pytorch_type[dtype]))


def full(g, sizes, value, dtype, layout, device):
    const_value = _maybe_get_const(value, 't')
    if _is_value(const_value):
        tmp = zeros(sizes, dtype, layout, device)
        return add(tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = _get_const(dtype, 'i', 'dtype')
        return g.op("ConstantOfShape", sizes,
                    value_t=torch.tensor(const_value, dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'f', 'i', 'v', 'v')
def full_like(g, input, fill_value, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor(fill_value, dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, self, dim, start, end, step):
    if step != 1:
        _unimplemented("slice", "step!=1 is currently not supported")
    if start.node().kind() != 'onnx::Constant' or \
            end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant':
        start_unsqueezed = g.op("Unsqueeze", start, axes_i=[0])
        end_unsqueezed = g.op("Unsqueeze", end, axes_i=[0])
        dim_unsqueezed = g.op("Unsqueeze", dim, axes_i=[0])
        return g.op("DynamicSlice", self, start_unsqueezed, end_unsqueezed, dim_unsqueezed)
    else:
        start = _parse_arg(start, 'i')
        end = _parse_arg(end, 'i')
        dim = _parse_arg(dim, 'i')
        return g.op("Slice", self, axes_i=[dim], starts_i=[start], ends_i=[end])


@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    return g.op("Clip", self, min_f=min_val, max_f=max_val)


def alias(g, self):
    return self


@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    return g.op("Unsqueeze", self, axes_i=[dim])


@parse_args('v', 'i', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")

    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


def to(g, self, *args):
    # ONNX doesn't have a concept of a device, so we ignore device casts
    if len(args) == 3:
        if args[0].type().isSubtypeOf(ListType.ofInts()):
            # aten::to(Tensor, Device, bool, bool)
            return self
        else:
            # aten::to(Tensor, ScalarType, bool, bool)
            dtype = _get_const(args[0], 'i', 'dtype')
            return g.op("Cast", self, to_i=scalar_type_to_onnx[dtype])
    elif len(args) == 4:
        # aten::to(Tensor, Device, ScalarType, bool, bool)
        dtype = _get_const(args[1], 'i', 'dtype')
        return g.op("Cast", self, to_i=scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool) -> Tensor
        dtype = _get_const(args[0], 'i', 'dtype')
        # Layout and device are ignored
        return g.op("Cast", self, to_i=scalar_type_to_onnx[dtype])
    else:
        raise NotImplementedError("Unknown aten::to signature")


def repeat(g, self, repeats):
    if not _is_value(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    const_repeats = _maybe_get_const(repeats, 'is')

    if self.isTensor() and not _is_value(const_repeats):
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
    after_view = view(g, self, [-1, upscale_factor, upscale_factor,
                                output_channel, dims[2], dims[3]])
    after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
    return view(g, after_transpose,
                [-1, output_channel, dims[2] * upscale_factor, dims[3] *
                 upscale_factor])


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    return g.op("ATen", input, weight, bias, num_groups_i=num_groups,
                eps_f=eps, cudnn_enabled_i=cudnn_enabled, operator_s="group_norm")


def _generic_rnn(g, variant, input, initial_states, all_weights, has_biases,
                 num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None):
    weights_per_layer = 4 if has_biases else 2
    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [all_weights[i:i + weights_per_layer] for i in range(0, len(all_weights), weights_per_layer)]
    if batch_first:
        return _unimplemented("RNN/GRU/LSTM", "batch_first")
    if dropout and train:
        return _unimplemented("RNN/GRU/LSTM", "dropout in training mode")

    if variant.startswith('RNN'):
        nonlinearity = variant[4:].lower()
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
        slices = [g.op('Slice', w, axes_i=[0], starts_i=[x * n], ends_i=[y * n]) for x, y in intervals]
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
        return x if num_layers == 1 else g.op('Slice', x, axes_i=[0], starts_i=[start], ends_i=[end])

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
            prev_output, h_out = g.op('RNN', *inputs, outputs=2,
                                      hidden_size_i=hidden_size,
                                      activations_s=[nonlinearity],
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
    h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
    if variant == 'RNN' or variant == 'GRU':
        return prev_output, h_outs
    elif variant == 'LSTM':
        c_outs = c_out if num_layers == 1 else g.op('Concat', *c_outs, axis_i=0)
        return prev_output, h_outs, c_outs


@parse_args('v', 'v', 'v', 'i', 'i', 'f', 'i', 'i', 'i')
def _lstm_full(g, input, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    hidden, weight = _unpack_list(hidden_v), _unpack_list(weight_v)
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers,
                        dropout, train, bidirectional, batch_first)


@parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
def _lstm_packed(g, input, batch_sizes, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional):
    hidden, weight = _unpack_list(hidden_v), _unpack_list(weight_v)
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers,
                        dropout, train, bidirectional, batch_sizes=batch_sizes)


def lstm(g, *args):
    if _is_tensor_list(args[3]):
        return _lstm_packed(g, *args)
    else:
        return _lstm_full(g, *args)


def _one_hidden_rnn(kind):
    @parse_args('v', 'v', 'v', 'i', 'i', 'f', 'i', 'i', 'i')
    def _rnn_full(g, input, hidden, weight_v, has_biases, num_layers, dropout, train, bidirectional, batch_first):
        weight = _unpack_list(weight_v)
        return _generic_rnn(g, kind, input, hidden, weight, has_biases, num_layers,
                            dropout, train, bidirectional, batch_first)

    @parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
    def _rnn_packed(g, input, batch_sizes, hidden, weight_v, has_biases, num_layers, dropout, train, bidirectional):
        weight = _unpack_list(weight_v)
        return _generic_rnn(g, kind, input, hidden, weight, has_biases, num_layers,
                            dropout, train, bidirectional, batch_sizes=batch_sizes)

    def symbolic(g, *args):
        if _is_tensor_list(args[3]):
            return _rnn_packed(g, *args)
        else:
            return _rnn_full(g, *args)

    return symbolic


gru = _one_hidden_rnn('GRU')
rnn_tanh = _one_hidden_rnn('RNN_TANH')
rnn_relu = _one_hidden_rnn('RNN_RELU')


@parse_args('v', 'i')
def _dim_arange(g, like, dim):
    return g.op('ATen', like, dim_i=dim, operator_s='_dim_arange')


def detach(g, input):
    # Erase aten::detach nodes because ONNX is inference only
    return input


def contiguous(g, input):
    return input


@parse_args('v', 'v', 'i')
def _pack_padded_sequence(g, input, lengths, batch_first):
    # There currently is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.
    if batch_first:
        input = g.op('Transpose', input, perm_i=[1, 0, 2])
    if not lengths.type().isSubtypeOf(torch._C.DynamicType.get()):
        raise RuntimeError("Lengths must be a Tensor for ONNX export")
    # We know it's a TensorType so this check is now safe.
    # It's really only necessary beacuse those operators expand to something that
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
    shape = _maybe_get_const(shapes_list[0], "is")
    return g.op('RandomNormal', shape_i=shape)


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
    if end_dim < 0 :
        end_dim = dim + end_dim
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1 and end_dim == dim - 1 :
        return g.op("Flatten", input, axis_i=start_dim)
    if start_dim == 0 and end_dim == dim - 2 :
        return g.op("Flatten", input, axis_i=end_dim + 1)
    # use Reshape for cases where the output shape is not 2D
    if input.type().kind() != "CompleteTensorType":
        return _unimplemented("flatten", "input size not accesible")
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
    return g.op('NonZero', input)
