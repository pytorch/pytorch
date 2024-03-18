import torch
import numpy as np
# aka torch.library
from library import Operator, traceable, inlinable, triton_kernel_call, triton_wrapper, Function, BackendImpl, def_blackbox, def_traceable
from torch import Tensor

# =====================================================================
# custom op API #3: def_blackbox / def_traceable
# =====================================================================

# =====================================================================
# To use PyTorch (and torch.compile) with a third-party library or
# custom kernel, use def_blackbox to create a black-box custom op.
#
# Morally, this represents some low-level kernel.
# PyTorch (e.g. torch.compile) will NEVER peek into a black-box op.

# We infer a schema from the type signature and from `mutable_args`,
# which is required. This suffices for most custom ops. A user can
# manually specify a schema if they need to.
@def_blackbox(mutable_args=[])
def numpy_sin(x: Tensor) -> Tensor:
    result = torch.from_numpy(np.sin(x.detach().cpu().numpy()))
    return result.to(x.device)

@numpy_sin.impl_abstract
def _(x):
    return torch.empty_like(x)

# =====================================================================
# Example of a mutable custom op.
@def_blackbox(mutable_args=["x"])
def my_sin_inplace(x: Tensor) -> None:
    x_np = x.detach().numpy()
    np.sin(x_np, out=x_np)

# ====================================================================
# Need to add autograd support? You can do it one of two ways
# 1. Wrap the op in an autograd.Function
# 2. Use impl_autograd
#
# Which one should you pick?
# - use autograd.Function if you're working with regular Tensors (i.e.
#   no Tensor subclasses).
# - use impl_autograd if you can guarantee that the autograd formula is
#   *traceable* and if any of the following apply:
#   - you want to add autograd support directly to the operator.

# ====================================================================
# Method 1: wrap a call to the custom op in an autograd.Function
class MySin(torch.autograd.Function):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return numpy_sin(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.saved_tensors[0].cos()


my_sin = MySin.apply

# ====================================================================
# Method 2: Use impl_autograd to directly add autograd support to
# the custom op.
def setup_context(ctx, args, output):
    x, = args
    ctx.save_for_backward(x)

def backward(ctx, grad_out):
    return grad_out * ctx.saved_tensors[0].cos()

numpy_sin.impl_autograd(setup_context, backward)


# =====================================================================
# Use `def_traceable` to define a custom op whose implementation is
# *traceable*: that is, the custom op's outputs are only determined by
# the inputs to the op and the op only consists of pytorch operations.
# If you're just using torch.compile, you generally shouldn't need to use this.
#
# The typical use case is using a different backend with torch.compile/export
# and wanting to group a sequence of pytorch ops together. We'll automatically
# generate implementations (e.g. the abstract impl) from the provided
# implementation.

@def_traceable(mutable_args=[])
def my_sin_cos(x: Tensor) -> Tensor:
    return x.sin().cos()

