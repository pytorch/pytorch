import torch
from torch._custom_function import CustomVjp, to_custom_vjp
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch import vmap, grad
import functools
import torch.utils._pytree as pytree
from torch._C import (
    DispatchKey,
)

mysum = PyOperator("mysum")

@mysum.py_functorch_impl(TransformType.Vmap)
def mysum_batch_rule(interpreter, x, dim):
    print("invoked")

    if not torch._C._functorch.is_batchedtensor(x): 
        with interpreter.lower():
            return mysum(x, dim)

    bdim = torch._C._functorch.maybe_get_bdim(x)
    value = torch._C._functorch.get_unwrapped(x)

    with interpreter.lower():
        value = value.movedim(bdim, 0)
        return mysum(value, dim + 1)

@mysum.py_impl(torch._C.DispatchKey.AutogradCPU)
def mysum_autograd(x, dim):
    return torch.sum(x, dim)


torch.manual_seed(0)
x = torch.randn(2, 3)
y = mysum(x, 1)
assert torch.allclose(y, x.sum(1))

def test(f, f_p, in_dims, args):
    expected = vmap(f, in_dims)(*args)
    result = vmap(f_p, in_dims)(*args)
    assert torch.allclose(result, expected)

# single vmap
test(torch.sum, mysum, (0, None), (x, 0))

# nested vmap
x = torch.randn(2, 3, 4)
test(vmap(functools.partial(torch.sum, dim=0)),
     vmap(functools.partial(mysum, dim=0)),
     (0,),
     (x,))



class MySin(CustomVjp):
    @staticmethod
    def forward(x):
        return x.sin(), [x]

    @staticmethod
    def backward(saved, grads):
        x, = saved
        gy, = grads
        return 2 * gy * x.cos()

x = torch.randn([], requires_grad=True)
y = MySin.apply(x)
gx, = torch.autograd.grad(y, x)
assert torch.allclose(gx, 2 * x.cos())

class MyMatmul(CustomVjp):
    @staticmethod
    def forward(x, y):
        return torch.mm(x, y), (x, y)

    @staticmethod
    def backward(saved, grads):
        gxy, = grads
        x, y = saved
        gx = gxy @ y.T
        gy = x.T @ gxy
        return 2 * gx, 2 * gy

x = torch.randn(3, 4, requires_grad=True)
y = torch.randn(4, 5, requires_grad=True)
gz = torch.randn(3, 5)

z = MyMatmul.apply(x, y)
gx, gy = torch.autograd.grad(z, (x, y), gz)
assert torch.allclose(gx, 2 * (gz @ y.T))
assert torch.allclose(gy, 2 * (x.T @ gz))

x = torch.randn(2, 3, 4, requires_grad=True)
y = torch.randn(2, 4, 5, requires_grad=True)

z = vmap(MyMatmul.apply)(x, y)
gx, gy = torch.autograd.grad(z, [x, y], torch.ones_like(z))

z = vmap(torch.mm)(x, y)
egx, egy = torch.autograd.grad(z, [x, y], torch.ones_like(z))
assert torch.allclose(gx / 2, egx)
assert torch.allclose(gy / 2, egy)

class MySin(CustomVjp):
    @staticmethod
    def forward(x):
        return x.sin(), [x]

    @staticmethod
    def backward(saved, grads):
        x, = saved
        gy, = grads
        return gy * x.sin()


x = torch.randn([])
y = grad(MySin.apply)(x)
assert torch.allclose(y, x.sin())

x = torch.randn([], requires_grad=True)
y = MySin.apply(x)
gx, = torch.autograd.grad(y, x, create_graph=True)
ggx, = torch.autograd.grad(gx, x)
assert torch.allclose(ggx, x.cos())

x = torch.randn([])
y = grad(grad(MySin.apply))(x)
assert torch.allclose(y, x.cos())

x = torch.randn(3)
y = vmap(grad(MySin.apply))(x)
assert torch.allclose(y, x.sin())

# Things to test:
#
# grad
# vmap
# vmap x grad
# grad x grad
# jacrev
#
# - saved {input, output, intermediate}
# - {1, 2+} x {inputs, outputs}
# - inplace operations inside body
# - returns view

# Interestingly, in JAX, they don't require gradient definition for intermediates.
class Cube(CustomVjp):
    @staticmethod
    def forward(x):
        three_x_sq = 3 * (x ** 2)
        return x ** 3, [three_x_sq]

    @staticmethod
    def backward(saved, grads):
        three_x_sq, = saved
        gy, = grads
        return three_x_sq * gy

x = torch.tensor(1., requires_grad=True)
gx = grad(Cube.apply)(x)
ggx = grad(grad(Cube.apply))(x)
print(gx)
print(ggx)


class MySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, gy):
        x, = ctx.saved_tensors
        return 2* gy * x.cos()


custom_sin = to_custom_vjp(MySin).apply
x = torch.randn([])
gx = grad(custom_sin)(x)
assert torch.allclose(gx, 2 * x.cos())

# import torch
# class MySquare(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         two_x = 2 * x
#         ctx.save_for_backward(two_x)
#         return x ** 2
# 
#     @staticmethod
#     def backward(ctx, gy):
#         two_x, = ctx.saved_tensors
#         return gy * two_x
# 
# x = torch.randn([], requires_grad=True)
# y = MySquare.apply(x)
# gy = torch.randn([], requires_grad=True)
# gx, = torch.autograd.grad(y, x, gy, create_graph=True)


import torch
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x ** 2
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gy):
        result, = ctx.saved_tensors
        return gy * 2 * result.sqrt()

x = torch.randn([], requires_grad=True)
y = MySquare.apply(x)
gy = torch.randn([], requires_grad=True)
gx, = torch.autograd.grad(y, x, create_graph=True)
ggx, = torch.autograd.grad(gx, x)
assert torch.allclose(ggx, torch.tensor(2.))

ggx = grad(grad(to_custom_vjp(MySquare).apply))(x)
assert torch.allclose(ggx, torch.tensor(2.))

import torch
import numpy as np
from torch._custom_function import to_custom_function
from torch.testing._internal.common_utils import disable_functorch


class NumpySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_np = x.detach().numpy()
        ctx.x_np = x_np
        return torch.tensor(np.sin(x_np))

    @staticmethod
    def backward(ctx, gy):
        # TODO: this shouldn't be necessary
        with disable_functorch():
            gx = gy.numpy() * np.cos(ctx.x_np)
        return torch.tensor(gx)

    @staticmethod
    def vmap_rule(ctx, x_batched):
        x, x_bdim = x_batched
        x_np = x.numpy()
        ctx.x_np = x_np
        return torch.tensor(np.sin(x_np)), x_bdim


def numpy_sin(x):
    output_and_ctx = to_custom_function(NumpySin)(x)
    return output_and_ctx.output


print('-' * 80)
x = torch.randn(3)
y = vmap(numpy_sin)(x)
assert torch.allclose(y, x.sin())

x = torch.randn([])
y = grad(numpy_sin)(x)
assert torch.allclose(y, x.cos())

y = numpy_sin(x)
assert torch.allclose(y, x.sin())

# expected to fail
# x = torch.randn([])
# y = grad(grad(numpy_sin))(x)
# assert torch.allclose(y, -x.sin())

# The composable autograd.function...
class BetterNumpySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print("forward")
        x_np = x.numpy()
        ctx.x_np = x_np
        return torch.tensor(np.sin(x_np)), x_np

    @staticmethod
    def backward(ctx, gy, _):
        print("backward")
        output_and_ctx = to_custom_function(NumpySinBackward)(gy, ctx.x_np)
        return output_and_ctx.output

    @staticmethod
    def vmap_rule(ctx, bx):
        x, x_bdim = bx
        x.movedim(x_bdim, 0)

        output_and_ctx = to_custom_function(BetterNumpySin)(x)
        y, x_np = output_and_ctx.output

        ctx.x_np = x_np
        return (y, 0), (x_np, None)


class NumpySinBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gy, x_np): 
        gx = gy.numpy() * np.cos(x_np)
        return torch.tensor(gx)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError("no double backwards support")

    @staticmethod
    def vmap_rule(ctx, bgy, b_x_np):
        gy, g_bdim = bgy
        x_np, _ = b_x_np  # TODO: passing convention
        return (NumpySinBackward.apply(gy, x_np), g_bdim)


def better_numpy_sin(x):
    output_and_ctx = to_custom_function(BetterNumpySin)(x)
    y, x_np = output_and_ctx.output
    return y

x = torch.randn(2, 3)
y = vmap(better_numpy_sin)(x)
assert torch.allclose(y, x.sin())

x = torch.randn(2, 3)
y = vmap(vmap(better_numpy_sin))(x)
assert torch.allclose(y, x.sin())

x = torch.randn([])
y = grad(better_numpy_sin)(x)
assert torch.allclose(y, x.cos())

print('*' * 80)
x = torch.randn(3)
y = vmap(grad(better_numpy_sin))(x)
assert torch.allclose(y, x.cos())


# Splitting this up
# 1. PyDispatcher supports functorch operations and mirrors functorch dispatch
# 2. vmap_rule
# 3. composable autograd.Function for functorch
#
# vmap_rule open questions:
# - calling conventions
# - captures, mutations, views (need test cases)
# We cannot support ctx.mark_dirty, until mode-only functorch (object identity...)
# views: should just work.
#
# composable autograd.Function for functorch open questions:
# - captures, mutations, views
# - API (decorator? flag?)
# - JAX difference flag
