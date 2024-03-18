import torch
import numpy as np
# aka torch.library
from library import Operator, traceable, inlinable, triton_kernel_call, triton_wrapper, Function, BackendImpl

# =====================================================================
# custom op API #2: BackendImpl and Function.
#
# In this version of the API, there are two main custom op APIs:
# - Use library.BackendImpl to define a black-box operation with an abstract impl
# - Use library.Function to define transform rules (e.g. autograd formula) for
# a sequence of make_fx-able operations (e.g. 1+ BackendImpl, existing
# PyTorch operators, or triton kernels)
#
# This approach is more explicit about traceability (Function's
# implementation MUST be traceable) and also allows automagic schema
# inference (because Function is always traceable, we can infer a schema
# for it).
# =====================================================================

# =====================================================================
# To use PyTorch (and torch.compile) with a third-party library or
# custom kernel, create a BackendImpl.
#
# Morally, this represents some low-level kernel and its abstract impl.
# PyTorch (e.g. torch.compile) will NEVER peek into BackendImpl -- it
# is always treated as a black-box.
#
# BackendImpl is not differentiable (or transformable, e.g. with functorch)
# by itself: to add an autograd formula or another transform rule, see below.
class NumpySin(BackendImpl):
    # Maybe we can infer this schema from the first call :P
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


# =====================================================================
# To specify custom functionality for a subsystem like autograd or vmap,
# a user should use Function (torch.library.Function).
#
# Functions should be functionally pure, e.g. their outputs are only
# determined from the inputs are there are no observable side effects.
#
# Under the hood, `Function` automatically constructs 1+ custom ops and
# uses Dispatch keys to add functionality.
#
# We require all staticmethods of `Function` be make_fx traceable:
# that is, torch.compile will peek into them. All `Function` will decompose
# into their forward() staticmethod when they hit Inductor.
#
# This is intentionally confusingly named like torch.autograd.Function,
# because it is the thing you should use instead of torch.autograd.Function :).
class MySin(Function):
    @staticmethod
    def forward(x):
        return NumpySin.apply(x)

    @staticmethod
    def post_forward(ctx, args, output):
        x, = args
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.saved_tensors[0].cos()


my_sin = MySin.apply

# =====================================================================
# A common use case is if a user already has a cuda kernel written in
# C++ and have bound it into python via the TORCH_LIBRARY API.
#
# Let's assume the user has a torch.ops.mycudalib.sin already.
#
# Morally, this operator is a "BackendImpl"; to add autograd (and other
# subsystem support), they should use Function:
#
# class MySin(Function):
#     @staticmethod
#     def forward(x):
#         return torch.ops.mycudalib.sin(x)
#
#     @staticmethod
#     def post_forward(ctx, args, output):
#         x, = args
#         ctx.save_for_backward(x)
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         return grad_out * ctx.saved_tensors[0].cos()


# =====================================================================
# Use case: people implementing their own backend may not wish for an
# sequence of PyTorch operations to decompose.
#
# That's simple to express: just provide a Function. No abstract impl
# necessary -- we automatically generate it from the forward.
class MySinCos(Function):
    # You can and should provide a schema if you care about export,
    # otherwise, we (PyTorch) will autogenerate something for you.
    schema = "mylib::my_sin_cos(Tensor x) -> Tensor"

    @staticmethod
    def forward(x):
        return x.sin().cos()

my_sin_cos = MySinCos.apply


# =====================================================================
# Mutable black-box op example
class MySinInplace(BackendImpl):
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


# =====================================================================
# triton custom op example
#
# The user wraps their Triton kernel in a `Function`: no need to use
# BackendImpl here because BackendImpls are black-boxes and we *want*
# torch.compile to poke at the triton kernel.
class MyAdd(Function):
    @staticmethod
    def forward(x, y):
        result = add_triton(x, y)
        return result

    def post_forward(ctx, args, output):
        pass

    def backward(ctx, grad_out):
        return grad_out, grad_out

my_add = MyAdd.apply


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


def add_triton(x, y):
    assert x.shape == y.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # add_kernel[(2,1,1)](x, y, 4, out, n_elements)
    # Wraps the triton kernel into a structured call that make_fx tracing understands.
    triton_kernel_call(add_kernel, (2,1,1), x, y, 4, out, n_elements)
    return out




# @torch.compile(backend="inductor")
# def f(x, y):
#     # return add_triton(x, y)
#     return my_add(x, y)
#
# x = torch.randn(5, device='cuda')
# y = torch.randn(5, device='cuda')
# z = f(x, y)
# assert torch.allclose(z, x + y)
