import torch
from torch.autograd._functions.utils import check_onnx_broadcast  # TODO: move me
from torch.nn.modules.utils import _pair, _triple
import warnings

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
    if isinstance(self, torch._C.Node):
        return self
    else:
        ty = tensor.type().scalarType().lower()
        return getattr(self, ty)()


def _broadcast_if_scalar(x):
    """Return kwargs enabling broadcasting if 'x' is a scalar."""
    if isinstance(x, torch._C.Node):
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


def addmm(g, self, mat1, mat2, beta, alpha):
    return g.op("Gemm", mat1, mat2, self, beta_f=_scalar(beta), alpha_f=_scalar(alpha))


def neg(g, self):
    return g.op("Neg", self)


def tanh(g, self):
    return g.op("Tanh", self)


def sigmoid(g, self):
    return g.op("Sigmoid", self)


def mean(g, self, dim=None, keepdim=None):
    kwargs = {}
    # NB: ONNX's default is different from PyTorch's
    if keepdim is None:
        keepdim = 0
    return g.op("ReduceMean", self, axes_i=dim, keepdims_i=keepdim)


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def expand(g, self, size):
    # TODO: This is not a real ONNX operator at the moment
    return g.op("Expand", self, shape_i=size)


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
    if self.type().sizes()[0] == size[0]:
        return g.op("Flatten", self, axis_i=1)
    return g.op("Reshape", self, shape_i=size)


def split(g, self, split_size, dim):
    size = self.type().sizes()[dim]
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=len(splits))


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
    return g.op('Softmax', input, axis_i=dim)


def softplus(g, self, beta, threshold):
    if beta != 1:
        return _unimplemented("beta", "has to be 1")
    return g.op('Softplus', self)


def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return _unimplemented("max_pool2d", "ceil_mode")
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d", "dilation")
    if not stride:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_pair(kernel_size),
             pads_i=_pair(padding),
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
                pads_i=_pair(padding))


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


def log_softmax(g, input, dim=None):
    return g.op("Log", g.op('Softmax', input, axis_i=dim).setTypeAs(input))


def unfold(g, input, dimension, size, step):
    return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)


def elu(g, input, alpha, inplace=False):
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=_scalar(alpha))


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
