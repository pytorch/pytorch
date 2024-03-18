import torch
import numpy as np
# aka torch.library
from library import Operator, traceable, inlinable, triton_kernel_call, triton_wrapper, Function

# =====================================================================
# custom op API #1: Operator API
#
# In this version of the API, there is a single Python custom op API:
# users can define an Operator and specify backend implementations for it,
# autograd formulas, and rules for other transforms (like vmap).
#
# In addition, users may use the @traceable / @inlinable decorators to
# say that "torch.compile should trace into the implementation". This
# is useful for when the e.g. cuda implementation contains a triton kernel:
# we want torch.compile to see the triton kernel and handle it.
# =====================================================================

# User provides their custom op schema and implementations
class MySin(Operator):
    schema = "(Tensor x) -> Tensor"

    # the black-box cpu kernel
    @staticmethod
    def impl_cpu(x):
        result = torch.from_numpy(np.sin(x.detach().cpu().numpy()))
        return result

    # the black-box cuda kernel
    @staticmethod
    def impl_cuda(x):
        return torch.from_numpy(np.sin(x.detach().cpu().numpy())).to(x.device)

    # the abstract impl. Must be "traceable". User must use opcheck to test.
    @staticmethod
    def abstract(x):
        return torch.empty_like(x)

    # autograd: provide us setup_backward() and backward() methods.
    # these must be "traceable". User must use opcheck to test.
    @staticmethod
    def setup_backward(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * x.cos()


# The user may optionally wrap in a function that provides a docstring and type hints.
def my_sin(x: torch.Tensor) -> torch.Tensor:
    """my_sin(x: Tensor) -> Tensor

    Returns the sin of x.
    """
    return MySin.call(x)


# Example of an operator that is implemented with pytorch operations
# We automatically generate an abstract impl for it.
class MySinCos(Operator):
    schema = "(Tensor x) -> Tensor"

    # Instead of specifying separate per-device impls, the user may give us a
    # single `impl` staticmethod that we will apply to all backends,
    # CompositeExplicitAutograd-style.
    @staticmethod
    # Specifies that the impl is make_fx traceable. We will autogenerate rules
    # (e.g. abstract, autograd, vmap). The user may override these by declaring
    # those methods.
    # This decorator may only be applied to `impl`.
    @traceable
    def impl(x):
        return x.sin().cos()


def my_sin_cos(x):
    """my_sin_cos(x: Tensor) -> Tensor

    Returns x.sin().cos()
    """
    return MySinCos.call(x)


# Mutable op example
class MySinInplace(Operator):
    schema = "(Tensor(a!) x) -> ()"

    # the black-box cpu kernel
    @staticmethod
    def impl_cpu(x):
        x_np = x.detach().numpy()
        np.sin(x_np, out=x_np)

    # the abstract impl. Must be "traceable". User must use opcheck to test.
    @staticmethod
    def abstract(x):
        return None


import triton
from triton import language as tl

@triton_wrapper
@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    BLOCK_SIZE: "tl.constexpr",
    out_ptr,
    n_elements,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


# @torch._dynamo.allow_in_graph
def add_triton(x, y):
    assert x.shape == y.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[(2,1,1)](x, y, 4, out, n_elements)
    # triton_kernel_call(add_kernel, (2,1,1), x, y, 4, out, n_elements)
    return out


class MyAdd(Operator):
    schema = "(Tensor x, Tensor y) -> Tensor"

    @staticmethod
    @inlinable(True/False) # user must provide this decorator
    def impl_cuda(x, y):
        # Problems: we don't end up using the schema! Maybe we should make_fx trace first
        result = add_triton(x, y)
        return result

    # Hmm, maybe "inlineable" should autogenerate this.
    # Not sure if it's too much magic
    # @staticmethod
    # def abstract(x, y):
    #     return torch.empty_like(x)

    def post_forward(ctx, args, output):
        pass

    def backward(ctx, grad_out):
        return grad_out, grad_out



# TODO: Delete
my_add = MyAdd.opoverload

# @torch.compile(backend="inductor")
# def f(x, y):
#     # return add_triton(x, y)
#     return my_add(x, y)
#
# x = torch.randn(5, device='cuda')
# y = torch.randn(5, device='cuda')
# z = f(x, y)
# assert torch.allclose(z, x + y)
