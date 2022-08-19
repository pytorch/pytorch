import torch
from functorch import vmap
from functorch._src.vmap import stupid_vmap
from torch.utils._pytree import *


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

    return new_f_fwd, new_f_bwd

# vmap over custom_vjp_call(f_fwd, f_bwd, *operands)
# x, y both batched
x = torch.randn(3)
y = torch.randn(3)
gxy = torch.randn(3)

new_f_fwd, new_f_bwd = batchify(f_fwd, f_bwd, (0, 0), 3)

xyy, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gxy, saved)

# vmap over custom_vjp_call(f_fwd, f_bwd, *operands)
# x batched, y not batched
x = torch.randn(3)
y = torch.randn([])
gxy = torch.randn(3)

new_f_fwd, new_f_bwd = batchify(f_fwd, f_bwd, (0, None), 3)

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

new_f_fwd, new_f_bwd = batchify(f_fwd, f_bwd, (0, 0), 2)
z, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gz, saved)

# Only x is batched
x = torch.randn(2, 3, 4)
y = torch.randn(4, 5)
gz = torch.randn(2, 3, 5)

new_f_fwd, new_f_bwd = batchify(f_fwd, f_bwd, (0, None), 2)
z, saved = new_f_fwd(x, y)
gx, gy = new_f_bwd(gz, saved)
