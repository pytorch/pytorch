import torch
from torch.autograd._functions.utils import check_onnx_broadcast  # TODO: move me
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils.rnn import PackedSequence
import warnings

import torch.onnx

# EDITING THIS FILE? READ THIS FIRST!
#
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
    return x[0]


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

_onnx_opset_version = 2


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
    return g.op("Undefined")


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


def expand(g, self, size):
    # TODO: This is not a real ONNX operator at the moment
    return g.op("Expand", self, shape_i=size)


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op("Gather", weight, indices)


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
    if self.type().sizes()[0] == size[0] and len(size) == 2:
        return g.op("Flatten", self, axis_i=1)
    return g.op("Reshape", self, shape_i=size)


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
                pads_i=_triple(padding))


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


def log_softmax(g, input, dim=None):
    return g.op("LogSoftmax", input, axis_i=dim)


def _convolution(g, input, weight, bias, stride, padding, dilation,
                 transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled):
    weight_size = weight.type().sizes()

    args = [input, weight]
    # ONNX only supports 1D bias
    if bias.node().kind() != "Undefined" and len(bias.type().sizes()) == 1:
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

    if bias.node().kind() != "Undefined" and len(bias.type().sizes()) != 1:
        return g.op("Add", n, bias, broadcast_i=1, axis_i=1)
    else:
        return n


def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               is_test_i=not training,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               consumed_inputs_i=(0, 0, 0, 1, 1),
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


def max(g, self, other):
    return g.op("Max", self, other)


def min(g, self, other):
    return g.op("Min", self, other)


def eq(g, self, other):
    return g.op("Equal", self, other)


def exp(g, self):
    return g.op("Exp", self)


def conv_tbc(g, input, weight, bias, pad):
    return g.op("ATen", input, weight, bias, operator_s="conv_tbc", pad_i=pad)


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
        if dropout:
            return _unimplemented("RNN", "dropout")
        if bidirectional:
            return _unimplemented("RNN", "bidirectional")

        prev_output = input
        h_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            weight_ih, weight_hh, bias_ih, bias_hh = all_weights[i]

            bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

            h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in]
            prev_output, h_out = g.op('RNN', *inputs, outputs=2,
                                      hidden_size_i=hidden_size,
                                      activations_s=[nonlinearity.lower()])
            h_outs.append(h_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        return prev_output, h_outs

    return symbolic


def LSTM_symbolic_builder(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, **kwargs):
    def symbolic(g, input, all_weights, h0_and_c0, batch_sizes):
        if batch_first:
            return _unimplemented("LSTM", "batch_first")
        if dropout:
            return _unimplemented("LSTM", "dropout")
        if bidirectional:
            return _unimplemented("LSTM", "bidirectional")

        h0, c0 = h0_and_c0

        prev_output = input
        h_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            # pytorch is input, forget, cell, output.
            # onnx is    input, output, forget, cell.
            weight_ih, weight_hh, bias_ih, bias_hh = \
                [reform_weights(g, w, hidden_size, [(0, 1), (3, 4), (1, 3)]) for w in all_weights[i]]

            bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

            h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])
            c_in = c0 if num_layers == 1 else g.op('Slice', c0, axes_i=[0], starts_i=[i], ends_i=[i + 1])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in, c_in]
            prev_output, h_out = g.op('LSTM', *inputs, outputs=2, hidden_size_i=hidden_size)
            h_outs.append(h_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        return prev_output, h_outs, None

    return symbolic


def GRU_symbolic_builder(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, **kwargs):
    def symbolic(g, input, all_weights, h0, batch_sizes):
        if batch_first:
            return _unimplemented("GRU", "batch_first")
        if dropout:
            return _unimplemented("GRU", "dropout")
        if bidirectional:
            return _unimplemented("GRU", "bidirectional")

        prev_output = input
        h_outs = []

        sequence_lens = unused(g) if batch_sizes is None else batch_sizes

        for i in range(num_layers):
            # pytorch is reset, input, hidden
            # onnx is    input, reset, hidden
            weight_ih, weight_hh, bias_ih, bias_hh = \
                [reform_weights(g, w, hidden_size, [(1, 2), (0, 1), (2, 3)]) for w in all_weights[i]]

            bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)

            h_in = h0 if num_layers == 1 else g.op('Slice', h0, axes_i=[0], starts_i=[i], ends_i=[i + 1])

            inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens, h_in]
            prev_output, h_out = g.op(
                'GRU', *inputs, outputs=2, hidden_size_i=hidden_size, linear_before_reset_i=1)
            h_outs.append(h_out)
        h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
        return prev_output, h_outs

    return symbolic
