import torch
import functorch
from functorch import vmap, grad
from torch._C import DispatchKey
from functorch.experimental.ops import PyOperator
from functorch._src.vmap import stupid_vmap, unwrap_batched, wrap_batched
from torch.utils._pytree import *
from functools import partial


def f_fwd(x, y):
    # x * y * y
    y_sq = y ** 2
    xy_sq = x * y ** 2
    return xy_sq, [x, y_sq]

def f_bwd(grads, saved):
    print("invoked f_bwd")
    gxy_sq = grads
    x, y_sq = saved
    gx = gxy_sq * y_sq
    gy_sq = gxy_sq * x
    gy = gy_sq * 2 * y_sq.sqrt()
    return gx, gy

def reductify_leaf(tensor, tensor_bdim, desired_bdim):
    if tensor_bdim is None and desired_bdim is None:
        return tensor
    if tensor_bdim is None and desired_bdim is not None:
        raise RuntimeError('NYI: A')
    if tensor_bdim is not None and desired_bdim is None:
        return tensor.sum(tensor_bdim)
    return tensor.movedim(tensor_bdim, desired_bdim)

def reductify(tensors, tensor_bdims, desired_bdims):
    tensors, spec = tree_flatten(tensors)
    tensor_bdims, _ = tree_flatten(tensor_bdims)
    desired_bdims, _ = tree_flatten(desired_bdims)

    result = [reductify_leaf(tensor, bdim, desired_bdim)
              for tensor, bdim, desired_bdim
              in zip(tensors, tensor_bdims, desired_bdims)]
    return tree_unflatten(result, spec)

def batchify(f_fwd, f_bwd, in_dims, batch_size):
    out_dims = None

    def new_f_fwd(*args):
        nonlocal out_dims
        outs, out_dims2 = stupid_vmap(f_fwd, in_dims, batch_size)(*args)
        out_dims = out_dims2
        return outs

    def new_f_bwd(grad_outs, saved):
        assert out_dims is not None
        grad_ins, grad_ins_dims = stupid_vmap(f_bwd, out_dims, batch_size)(grad_outs, saved)
        return reductify(grad_ins, grad_ins_dims, in_dims)

    def get_out_dims():
        assert out_dims is not None
        return out_dims

    return new_f_fwd, new_f_bwd, get_out_dims

# vmap over custom_vjp_call(f_fwd, f_bwd, *operands)
# x, y both batched
x = torch.randn(3)
y = torch.randn(3)
gxy = torch.randn(3)

new_f_fwd, new_f_bwd, _ = batchify(f_fwd, f_bwd, (0, 0), 3)

xyy, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gxy, saved)

# vmap over custom_vjp_call(f_fwd, f_bwd, *operands)
# x batched, y not batched
x = torch.randn(3)
y = torch.randn([])
gxy = torch.randn(3)

new_f_fwd, new_f_bwd, _ = batchify(f_fwd, f_bwd, (0, None), 3)

xyy, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gxy, saved)

# OK, let's pick some non-trivial f_fwd / f_bwd
def f_fwd(x, y):
    return torch.mm(x, y), [x, y]

def f_bwd(gxy, saved):
    x, y = saved
    gx = gxy @ y.T
    gy = x.T @ gxy
    return gx, gy

# Both x and y are batched
x = torch.randn(2, 3, 4)
y = torch.randn(2, 4, 5)
gz = torch.randn(2, 3, 5)

new_f_fwd, new_f_bwd, _ = batchify(f_fwd, f_bwd, (0, 0), 2)
z, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gz, saved)

# Only x is batched
x = torch.randn(2, 3, 4)
y = torch.randn(4, 5)
gz = torch.randn(2, 3, 5)

new_f_fwd, new_f_bwd, _ = batchify(f_fwd, f_bwd, (0, None), 2)
z, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gz, saved)

def summ_vmap(x, dim):
    current_level = functorch._C.current_level()
    x, x_bdim = unwrap_batched(x)
    assert x_bdim == 0

    guard = functorch._C.WithoutTop()
    result = x.sum(dim + 1)
    del guard
    result = wrap_batched(current_level, result, x_bdim)
    return result

summ = PyOperator('summ')
summ.impl('vmap', summ_vmap)
x = torch.randn(2, 3, 4)
result = vmap(summ, (0, None))(x, 0)
assert torch.allclose(result, x.sum(1))
result = vmap(vmap(summ, (0, None)), (0, None))(x, 0)
assert torch.allclose(result, x.sum(2))


def custom_vjp_call_vmap(f_fwd, f_bwd, *operands):
    current_level = functorch._C.current_level()
    unwrapped_operands, in_dims = unwrap_batched(operands)
    new_f_fwd, new_f_bwd, get_out_dims = batchify(f_fwd, f_bwd, in_dims, 2)

    guard = functorch._C.WithoutTop()
    print("custom_vjp batch rule")
    result = custom_vjp_call(new_f_fwd, new_f_bwd, *unwrapped_operands)
    del guard

    out_dims = get_out_dims()
    return wrap_batched(current_level, result, out_dims)

def custom_vjp_call_autograd(f_fwd, f_bwd, *operands):
    # TODO:
    class Something(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            print("Something forward")
            outs, saved = f_fwd(*operands)
            ctx.save_for_backward(*saved)
            return outs, saved

        @staticmethod
        def backward(ctx, *grads):
            # TODO: account for `saved`
            grads = grads[:-1]
            print("Something backward")
            saved = ctx.saved_tensors
            if len(grads) == 1:
                grads = grads[0]
            result = f_bwd(grads, saved)
            return result
    return Something.apply(*operands)

custom_vjp_call = PyOperator('custom_vjp_call')
custom_vjp_call.impl('vmap', custom_vjp_call_vmap)
custom_vjp_call.impl(DispatchKey.AutogradCPU, custom_vjp_call_autograd)


def f(x, y):
    return custom_vjp_call(f_fwd, f_bwd, x, y)[0]

# I believe we have re-entrancy problems with vmap (invoking vmap from a batching rule)
x = torch.randn(2, 3, 4, requires_grad=True)
y = torch.randn(2, 4, 5, requires_grad=True)

print("begin vmap")
z = vmap(f)(x, y)
z.sum().backward()
print("end vmap")

# nested vmap
x = torch.randn(2, 2, 3, 4, requires_grad=True)
y = torch.randn(2, 2, 4, 5, requires_grad=True)
z = vmap(vmap(f))(x, y)
z.sum().backward()

def unwrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return functorch._C._unwrap_for_grad(t, level)
    return t

def wrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return functorch._C._wrap_for_grad(t, level)
    return t

# OK: can we do a grad rule for custom_vjp_call ??
def custom_vjp_call_grad(f_fwd, f_bwd, *operands):
    class Something(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            level = functorch._C.current_level()
            unwrapped_operands = tree_map(partial(unwrap_grad, level), operands)

            with torch.enable_grad():
                guard = functorch._C.WithoutTop()
                output = custom_vjp_call(f_fwd, f_bwd, *unwrapped_operands)
                results, saved = output
                ctx.save_for_backward(*saved)
                del guard

            results = tree_map(partial(wrap_grad, level), results)
            saved = tree_map(partial(wrap_grad, level), saved)
            return results, saved

        @staticmethod
        def backward(ctx, *grads):
            # TODO: accouunt for `saved`
            grads = grads[:-1]
            outs = f_bwd(grads, ctx.saved_tensors)
            return outs
    return Something.apply(*operands)

custom_vjp_call.impl('grad', custom_vjp_call_grad)

def f_fwd3(x):
    return x.sin(), [x]

def f_bwd3(gy, saved):
    x, = saved
    return gy[0] * x.sin()

def f(x):
    out = custom_vjp_call(f_fwd3, f_bwd3, x)
    return out[0]

x = torch.randn([])
y = grad(f)(x)
assert torch.allclose(y, x.sin())

x = torch.randn(3)
y = vmap(grad(f))(x)
assert torch.allclose(y, x.sin())
