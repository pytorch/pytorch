import numbers

import torch
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils.rnn import PackedSequence
import warnings

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from collections import Iterable
from functools import partial
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

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self
    else:
        ty = tensor.type().scalarType().lower()
        return getattr(self, ty)()


def _broadcast_if_scalar(x):
    """Return kwargs enabling broadcasting if 'x' is a scalar."""
    if isinstance(x, torch._C.Value):
        return {}
    else:
        return {"broadcast_i": 1}


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


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

_onnx_opset_version = 6


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
#   - Most of the time, the arguments to self/other are pre-expanded according
#     to broadcasting.  However, a scalar will NOT be broadcasted, so we have
#     to enable broadcasting ONNX side.
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
    return g.op("prim::Undefined")


def add(g, self, other, alpha):
    if _scalar(alpha) != 1:
        return _unimplemented("add", "alpha != 1")
    # See Note [Pointwise by scalar]
    return g.op("Add", self, _if_scalar_type_as(other, self), **_broadcast_if_scalar(other))


def sub(g, self, other, alpha):
    if _scalar(alpha) != 1:
        return _unimplemented("sub", "alpha != 1")
    # See Note [Pointwise by scalar]
    return g.op("Sub", self, _if_scalar_type_as(other, self), **_broadcast_if_scalar(other))


def mul(g, self, other):
    # See Note [Pointwise by scalar]
    return g.op("Mul", self, _if_scalar_type_as(other, self), **_broadcast_if_scalar(other))


def div(g, self, other):
    # See Note [Pointwise by scalar]
    return g.op("Div", self, _if_scalar_type_as(other, self), **_broadcast_if_scalar(other))


# This syntax is Python 2 portable
def cat(g, *tensors, **kwargs):
    dim = kwargs.pop("dim")
    assert not kwargs
    return g.op("Concat", *tensors, axis_i=dim)


def mm(g, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    ty = self.type().scalarType().lower()
    C = g.constant(0, [1], ty)
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0, broadcast_i=True)


def bmm(g, self, other):
    return g.op("MatMul", self, other)


def matmul(g, self, other):
    return g.op("MatMul", self, other)


def addmm(g, self, mat1, mat2, beta, alpha):
    return g.op("Gemm", mat1, mat2, self, beta_f=_scalar(beta), alpha_f=_scalar(alpha))


def neg(g, self):
    return g.op("Neg", self)


def sqrt(g, self):
    return g.op("Sqrt", self)


def tanh(g, self):
    return g.op("Tanh", self)


def sigmoid(g, self):
    return g.op("Sigmoid", self)


def mean(g, self, dim=None, keepdim=None):
    if dim is None and keepdim is None:
        return g.op("Mean", self)
    # NB: ONNX's default is different from PyTorch's
    if keepdim is None:
        keepdim = 0
    return g.op("ReduceMean", self, axes_i=[dim], keepdims_i=keepdim)


def sum(g, self, dim=None, keepdim=None):
    if dim is None and keepdim is None:
        return g.op("Sum", self)
    if keepdim is None:
        keepdim = 0
    return g.op("ReduceSum", self, axes_i=[dim], keepdims_i=keepdim)


def cumsum(g, input, dim):
    return g.op("ATen", input, operator_s="cumsum", dim_i=dim)


def prod(g, self, dim=None, keepdim=None):
    if dim is None:
        dims = None
    else:
        dims = [dim]
    if keepdim is None:
        keepdim = 0
    return g.op("ReduceProd", self, axes_i=dims, keepdims_i=keepdim)


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


# There is no translation for it, but we don't want to raise an error yet
def expand(g, self, size):
    return None


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op("Gather", weight, indices)


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
                outputs=3,
                scale_grad_by_freq_i=scale_grad_by_freq,
                mode_i=mode,
                sparse_i=sparse)


def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    axes = list(range(len(self.type().sizes())))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return g.op("Transpose", self, perm_i=axes)


def permute(g, self, dims):
    if dims == list(range(0, len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


def view(g, self, size):
    self_sizes = self.type().sizes()
    if self_sizes and len(size) == 2 and self_sizes[0] == size[0]:
        return g.op("Flatten", self, axis_i=1)
    shape = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Reshape", self, shape)


def split(g, self, split_size, dim):
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
def chunk(g, self, chunks, dim):
    split_size = (self.type().sizes()[dim] + chunks - 1) // chunks
    return split(g, self, split_size, dim)


def select(g, self, dim, index):
    slice_node = g.op("Slice", self, axes_i=[dim], starts_i=[index], ends_i=[index + 1])
    return g.op("Squeeze", slice_node, axes_i=[dim])


def squeeze(g, self, dim=None):
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [dim]
    return g.op("Squeeze", self, axes_i=dims)


def prelu(g, self, weight):
    return g.op("PRelu", self, weight)


def relu(g, input):
    return g.op("Relu", input)


def threshold(g, self, threshold, value):
    # See Note [Export inplace]
    if _scalar(threshold) != 0:
        return _unimplemented("threshold", "non-zero threshold")
    if _scalar(value) != 0:
        return _unimplemented("threshold", "non-zero value")
    return g.op("Relu", self)


def leaky_relu(g, input, negative_slope, inplace=False):
    # See Note [Export inplace]
    # TODO: Talk to ONNX about unconditional cast of scalar to float
    return g.op("LeakyRelu", input, alpha_f=_scalar(negative_slope))


def glu(g, input, dim):
    assert input.type().sizes()[dim] % 2 == 0

    first, second = g.op('Split', input, axis_i=dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))


def softmax(g, input, dim=None):
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
        dim = len(input.type().sizes()) + dim
    if len(input.type().sizes()) != dim + 1:
        return _unimplemented("dim", "ONNX and PyTorch use different strategies to split the input.")
    return g.op('Softmax', input, axis_i=dim)


def softplus(g, self, beta, threshold):
    if beta != 1:
        return _unimplemented("beta", "has to be 1")
    return g.op('Softplus', self)


def max_pool1d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool1d", "ceil_mode")
    if set(_single(dilation)) != {1}:
        return _unimplemented("max_pool1d", "dilation")
    if stride is None:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_single(kernel_size),
             pads_i=_single(padding) * 2,
             strides_i=_single(stride))
    return r, None


def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool2d", "ceil_mode")
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d", "dilation")
    if not stride:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_pair(kernel_size),
             pads_i=_pair(padding) * 2,
             strides_i=_pair(stride))
    return r, None


def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if ceil_mode:
        return _unimplemented("avg_pool2d", "ceil_mode")
    if not stride:
        stride = kernel_size
    # TODO: What about count_include_pad?!
    return g.op("AveragePool", input,
                kernel_shape_i=_pair(kernel_size),
                strides_i=_pair(stride),
                pads_i=_pair(padding) * 2)


def avg_pool3d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if ceil_mode:
        return _unimplemented("avg_pool3d", "ceil_mode")
    if not stride:
        stride = kernel_size
    # TODO: What about count_include_pad?!
    return g.op("AveragePool", input,
                kernel_shape_i=_triple(kernel_size),
                strides_i=_triple(stride),
                pads_i=_triple(padding) * 2)


def reflection_pad(g, input, padding):
    from torch.autograd._functions.utils import prepare_onnx_paddings
    mode = "reflect"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


def replication_pad(g, input, padding):
    from torch.autograd._functions.utils import prepare_onnx_paddings
    mode = "edge"
    paddings = prepare_onnx_paddings(len(input.type().sizes()), padding)
    return g.op("Pad", input, pads_i=paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def upsample_nearest2d(g, input, scale_factor):
    return g.op("Upsample", input, width_scale_f=scale_factor,
                height_scale_f=scale_factor, mode_s="nearest")


def upsample_bilinear2d(g, input, output_size):
    w_scale = float(output_size[-1]) / input.type().sizes()[-1]
    h_scale = float(output_size[-2]) / input.type().sizes()[-2]
    return g.op("Upsample", input, width_scale_f=w_scale,
                height_scale_f=h_scale, mode_s="bilinear")


def gt(g, input, other):
    return g.op("Greater", input, _if_scalar_type_as(other, input), **_broadcast_if_scalar(other))


def lt(g, input, other):
    return g.op("Less", input, _if_scalar_type_as(other, input), **_broadcast_if_scalar(other))


def log_softmax(g, input, dim=None):
    return g.op("LogSoftmax", input, axis_i=dim)


def _convolution(g, input, weight, bias, stride, padding, dilation,
                 transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled):
    weight_size = weight.type().sizes()

    args = [input, weight]
    # ONNX only supports 1D bias
    if bias.node().kind() != "prim::Undefined" and len(bias.type().sizes()) == 1:
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

    if bias.node().kind() != "prim::Undefined" and len(bias.type().sizes()) != 1:
        return g.op("Add", n, bias, broadcast_i=1, axis_i=1)
    else:
        return n


def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               is_test_i=not training,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               outputs=1 if not training else 5)
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setUniqueName("batch_norm_dead_output-" + saved_mean.uniqueName())
        saved_var.setUniqueName("batch_norm_dead_output-" + saved_var.uniqueName())
        return res


def unfold(g, input, dimension, size, step):
    return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)


def elu(g, input, alpha, inplace=False):
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=_scalar(alpha))


def selu(g, input):
    return g.op("Selu", input)


def index_select(g, self, index, dim):
    return g.op("Gather", self, index, axis_i=dim)


def type_as(g, self, other):
    if self.type().scalarType() == other.type().scalarType():
        # no-op
        return self
    else:
        # TODO: This should be pretty easy, just implement it with Cast
        return _unimplemented("type_as", "non no-op application")


# ignore clone operators that are inserted by PyTorch autograd
def clone(g, input):
    return input


def abs(g, self):
    return g.op("Abs", self)


def pow(g, self, exponent):
    return g.op("Pow", self, exponent)


def clamp(g, self, min, max):
    return g.op("Clip", self, min_f=min, max_f=max)


# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
def max(g, self, *args, **kwargs):
    dim = kwargs.get("dim", None)
    if dim is None and isinstance(args[0], numbers.Number):
        dim = args[0]
    if dim is not None:
        keepdim = kwargs.get("keepdim", False)
        # TODO: export it as ReduceMax
        return g.op("ATen",
                    self,
                    operator_s="max",
                    dim_i=dim,
                    keepdim_i=keepdim,
                    outputs=2)
    else:
        (other,) = args
        return g.op("Max", self, other)


def min(g, self, *args, **kwargs):
    dim = kwargs.get("dim", None)
    if dim is None and isinstance(args[0], numbers.Number):
        dim = args[0]
    if dim is not None:
        keepdim = kwargs.get("keepdim", False)
        # TODO: export it as ReduceMin
        return g.op("ATen",
                    self,
                    operator_s="min",
                    dim_i=dim,
                    keepdim_i=keepdim,
                    outputs=2)
    else:
        (other,) = args
        return g.op("Min", self, other)


def eq(g, self, other):
    return g.op("Equal", self, other)


def exp(g, self):
    return g.op("Exp", self)


def conv_tbc(g, input, weight, bias, pad):
    return g.op("ATen", input, weight, bias, operator_s="conv_tbc", pad_i=pad)


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
    'uint8_t': torch.onnx.TensorProtoDataType.UINT8,
    'int8_t': torch.onnx.TensorProtoDataType.INT8,
    'double': torch.onnx.TensorProtoDataType.DOUBLE,
    'float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'int': torch.onnx.TensorProtoDataType.INT32,
    'int64_t': torch.onnx.TensorProtoDataType.INT64,
    'int16_t': torch.onnx.TensorProtoDataType.INT16,
}


def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


for k, v in cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = partial(_cast_func_template, v)


def slice(g, self, dim, start, end, step):
    if step != 1:
        _unimplemented("slice", "step!=1 is currently not supported")
    return g.op("Slice", self, axes_i=[dim], starts_i=[start], ends_i=[end])


def alias(g, self):
    return self


def unsqueeze(g, self, dim):
    return g.op("Unsqueeze", self, axes_i=[dim])


def topk(g, self, k, dim=None, largest=True, sorted=True, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")

    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


def instance_norm(g, input, **kwargs):
    input_type = input.type().scalarType()
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    eps = kwargs.get("eps", 1e-5)
    if weight is None:
        weight = g.constant(1.0, [input.type().sizes()[1]], input_type)
    else:
        weight = g.op('Constant', value_t=weight)
    if bias is None:
        bias = g.constant(0.0, [input.type().sizes()[1]], input_type)
    else:
        bias = g.op('Constant', value_t=bias)
    return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)


def RNN_symbolic_builder(cell_type, *args, **kwargs):
    if cell_type == 'LSTM':
        return LSTM_symbolic_builder(*args, **kwargs)
    elif cell_type == 'GRU':
        return GRU_symbolic_builder(*args, **kwargs)
    elif cell_type.startswith('RNN_'):
        return Elman_RNN_symbolic_builder(cell_type[4:], *args, **kwargs)
    else:
        return lambda *args, **kwargs: _unimplemented("RNN", "cell type " + cell_type)


def reform_weights(g, w, n, intervals):
    slices = [g.op('Slice', w, axes_i=[0], starts_i=[x * n], ends_i=[y * n]) for x, y in intervals]
    return g.op('Concat', *slices, axis_i=0)


def Elman_RNN_symbolic_builder(
        nonlinearity, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, **kwargs):
    def symbolic(g, input, all_weights, h0, batch_sizes):
        if batch_first:
            return _unimplemented("RNN", "batch_first")
        if dropout and kwargs['train']:
            return _unimplemented("RNN", "dropout in training mode")

        unidirectional = not bidirectional

        prev_output = input
        h_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            if unidirectional:
                weight_ih, weight_hh, bias_ih, bias_hh = all_weights[i]
                bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])
            else:
                weight_ih = g.op('Concat', all_weights[2 * i][0], all_weights[2 * i + 1][0], axis_i=0)
                weight_hh = g.op('Concat', all_weights[2 * i][1], all_weights[2 * i + 1][1], axis_i=0)
                bias_concat = g.op('Concat',
                                   all_weights[2 * i][2],
                                   all_weights[2 * i][3],
                                   all_weights[2 * i + 1][2],
                                   all_weights[2 * i + 1][3],
                                   axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[2 * i], ends_i=[2 * i + 2])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in]
            extra_kwargs = {} if unidirectional else {'direction_s': 'bidirectional'}
            prev_output, h_out = g.op('RNN', *inputs, outputs=2,
                                      hidden_size_i=hidden_size,
                                      activations_s=[nonlinearity.lower()],
                                      **extra_kwargs)
            h_outs.append(h_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        return prev_output, h_outs

    return symbolic


def LSTM_symbolic_builder(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, **kwargs):
    def symbolic(g, input, all_weights, h0_and_c0, batch_sizes):
        if batch_first:
            return _unimplemented("LSTM", "batch_first")
        if dropout and kwargs['train']:
            return _unimplemented("RNN", "dropout in training mode")

        unidirectional = not bidirectional

        h0, c0 = h0_and_c0

        prev_output = input
        h_outs = []
        c_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            if unidirectional:
                # pytorch is input, forget, cell, output.
                # onnx is    input, output, forget, cell.
                weight_ih, weight_hh, bias_ih, bias_hh = \
                    [reform_weights(g, w, hidden_size, [(0, 1), (3, 4), (1, 3)]) for w in all_weights[i]]

                bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])
                c_in = c0 if num_layers == 1 else g.op('Slice', c0, axes_i=[0], starts_i=[i], ends_i=[i + 1])
            else:
                # pytorch is input, forget, cell, output.
                # onnx is    input, output, forget, cell.
                weight_ih_f, weight_hh_f, bias_ih_f, bias_hh_f = \
                    [reform_weights(g, w, hidden_size, [(0, 1), (3, 4), (1, 3)]) for w in all_weights[2 * i]]
                weight_ih_b, weight_hh_b, bias_ih_b, bias_hh_b = \
                    [reform_weights(g, w, hidden_size, [(0, 1), (3, 4), (1, 3)]) for w in all_weights[2 * i + 1]]

                weight_ih = g.op('Concat', weight_ih_f, weight_ih_b, axis_i=0)
                weight_hh = g.op('Concat', weight_hh_f, weight_hh_b, axis_i=0)
                bias_concat = g.op('Concat', bias_ih_f, bias_hh_f, bias_ih_b, bias_hh_b, axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[2 * i], ends_i=[2 * i + 2])
                c_in = c0 if num_layers == 1 else g.op('Slice', c0, axes_i=[0], starts_i=[2 * i], ends_i=[2 * i + 2])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in, c_in]
            extra_kwargs = {} if unidirectional else {'direction_s': 'bidirectional'}
            prev_output, h_out, c_out = g.op('LSTM', *inputs, outputs=3,
                                             hidden_size_i=hidden_size,
                                             **extra_kwargs)
            h_outs.append(h_out)
            c_outs.append(c_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        c_outs = c_out if num_layers == 1 else g.op('Concat', *c_outs, axis_i=0)
        return prev_output, h_outs, c_outs

    return symbolic


def GRU_symbolic_builder(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, **kwargs):
    def symbolic(g, input, all_weights, h0, batch_sizes):
        if batch_first:
            return _unimplemented("GRU", "batch_first")
        if dropout and kwargs['train']:
            return _unimplemented("RNN", "dropout in training mode")

        unidirectional = not bidirectional

        prev_output = input
        h_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            if unidirectional:
                # pytorch is reset, input, hidden
                # onnx is    input, reset, hidden
                weight_ih, weight_hh, bias_ih, bias_hh = \
                    [reform_weights(g, w, hidden_size, [(1, 2), (0, 1), (2, 3)]) for w in all_weights[i]]

                bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])
            else:
                # pytorch is reset, input, hidden
                # onnx is    input, reset, hidden
                weight_ih_f, weight_hh_f, bias_ih_f, bias_hh_f = \
                    [reform_weights(g, w, hidden_size, [(1, 2), (0, 1), (2, 3)]) for w in all_weights[2 * i]]
                weight_ih_b, weight_hh_b, bias_ih_b, bias_hh_b = \
                    [reform_weights(g, w, hidden_size, [(1, 2), (0, 1), (2, 3)]) for w in all_weights[2 * i + 1]]

                weight_ih = g.op('Concat', weight_ih_f, weight_ih_b, axis_i=0)
                weight_hh = g.op('Concat', weight_hh_f, weight_hh_b, axis_i=0)
                bias_concat = g.op('Concat', bias_ih_f, bias_hh_f, bias_ih_b, bias_hh_b, axis_i=0)

                h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[2 * i], ends_i=[2 * i + 2])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in]
            extra_kwargs = {} if unidirectional else {'direction_s': 'bidirectional'}
            prev_output, h_out = g.op('GRU', *inputs, outputs=2,
                                      hidden_size_i=hidden_size, linear_before_reset_i=1,
                                      **extra_kwargs)
            h_outs.append(h_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        return prev_output, h_outs

    return symbolic


# WARNING: Here be dragons. i.e. this is a hack that should die in a fire
#
# Since we need RNN nodes to work both in the GraphExecutor as well as call the
# correct symbolic function during ONNX export, we do the following:
#
# 1. During tracing we dispatch to this function
# 2. This function emits a PythonOp wrapping the RNN function that would have
#    run had we not been tracing. Thus, GraphExecutor will call the RNN operator
#    via Python. In the future we will likely want to make the RNN modules into
#    ScriptModules so we can optimize them.
# 3. We store a wrapper around the ONNX symbolic function in the `symbolic`
#    attribute of the Python function. The ONNX export pass accesses this
#    attribute during tracing and calls it to lower the PythonOp into the right
#    thing
#
# The first three parameters to this function are meant to be bound with:
#   cell_type - The string description of the type of RNN cell. e.g. 'LSTM'
#   func - The function that would have been called here if we had not been
#          tracing, e.g. CudnnRNN or AutogradRNN.
#   sym - The ONNX symbolic we should store in the PythonOp for later export.
#
# With those three parameters bound, we can pass the function into the
# torch.onnx.symbolic_override* functions
#
# The remaining arguments are equivalent to the inputs seen when dispatching
# a symbolic function for an operator. Concretely:
#  * input - a single input tensor [seq_len, batch, input_size] or if bach_first=True,
#            [batch, seq_len, input_size]
#  * weights - list of list of tensors. len(weights) = number of layers
#              weights[i] is a list of weights, same as the parameters to
#              torch.nn.{RNN,LSTM,GRU}. See the symbolic builders above
#  * hiddens - hidden state for the first layer, or {hidden state, cell state} if
#              cell_type == LSTM
#  * batch_sizes - 1-D tensor containing the sequence length for each example
#                  in the batch.
def rnn_trace_override_symbolic(cell_type, func, sym, g, input, weights, hiddens, batch_sizes):
    num_layers = len(weights)
    num_weights = 0
    for x in weights:
        num_weights += len(x)
    weights_per_layer = num_weights // num_layers
    has_batch_sizes = batch_sizes is not None

    # Since we need flat argument lists in the IR, these two functions and the
    # supporting code before the `wrapPyFuncWithSymbolic` call are simply
    # helpers to reconstruct the input, weights, hiddens, and batch_sizes
    # inputs from the flat argument list. To do this, the above code captures
    # then lengths of each of these inputs so that we can rematerialize them
    # later before calling either the RNN function or the ONNX symbolic function

    def forward_flattened_wrapper(input, *args):
        args_offset = 0
        weights = []
        for _ in range(num_layers):
            weights.append(args[args_offset:args_offset + weights_per_layer])
            args_offset += weights_per_layer
        if has_batch_sizes:
            hiddens = args[args_offset:-1]
            batch_sizes = args[-1]
        else:
            hiddens = args[args_offset:]
            batch_sizes = None
        if cell_type != 'LSTM':
            assert len(hiddens) == 1
            hiddens = hiddens[0]
        outputs = func(input, weights, hiddens, batch_sizes)
        # We also need a flattened output list
        outs_flattened = [outputs[0]]
        if cell_type == 'LSTM':
            for o in outputs[1]:
                outs_flattened.append(o)
        else:
            outs_flattened.append(outputs[1])
        return tuple(outs_flattened)

    def symbolic_flattened_wrapper(g, input, *args):
        args_offset = 0
        weights = []
        for _ in range(num_layers):
            weights.append(args[args_offset:args_offset + weights_per_layer])
            args_offset += weights_per_layer
        if has_batch_sizes:
            hiddens = args[args_offset:-1]
            batch_sizes = args[-1]
        else:
            hiddens = args[args_offset:]
            batch_sizes = None
        if cell_type != 'LSTM':
            assert len(hiddens) == 1
            hiddens = hiddens[0]
        return sym(g, input, weights, hiddens, batch_sizes)

    flattened_weights = []
    for x in weights:
        for y in x:
            flattened_weights.append(y)
    if not isinstance(hiddens, Iterable):
        hiddens = [hiddens]
    inputs = list(itertools.chain.from_iterable(
        [[input], flattened_weights, hiddens,
            [batch_sizes] if batch_sizes else []]))
    outputs = g.wrapPyFuncWithSymbolic(
        forward_flattened_wrapper,
        inputs,
        3 if cell_type == 'LSTM' else 2,
        symbolic_flattened_wrapper
    )
    return tuple(o for o in outputs)
