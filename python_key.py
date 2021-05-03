import time
import torch
import functools
from string import ascii_lowercase
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from types import FunctionType, CodeType
import functorch
from functorch import wrap_key, WrapModule, jacrev, vmap, grad, pythonkey_trace

torch._C._debug_only_display_vmap_fallback_warnings(True)


import torchvision.models as models

def fetch_attr(mod, target : str):
    target_atoms = target.split('.')
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

class ModuleBackward(nn.Module):
    def __init__(self, mod, inps):
        super().__init__()
        self.mod = mod
        self.inps = inps

    def f_grad(self, args):
        for idx in range(len(self.inps)):
            args[idx] = addPythonKey(PythonTensor(self.inps[idx].shape, args[idx]))
            # args[idx].requires_grad = False
        out = self.mod(*args)
        out.backward()
        # import pdb; pdb.set_trace()
        # return tuple(removePythonKey(p.grad) for p in self.parameters())
        param_grads = []
        for k,v in self.state_dict().items():
            val = fetch_attr(self, k)
            # param_grads[k] = key.removeKey(val.grad).proxy
            param_grads.append(key.removeKey(val.grad).proxy)
        return tuple(param_grads)
        # return tuple(key.removeKey(args[idx].grad).proxy for idx in range(len(inps)))

    def forward(self, a):
        return self.f_grad([a])

class Foo(torch.nn.Module):
    def __init__(self, num_features=50):
        super(Foo, self).__init__()
        self.linear = nn.Conv2d(3, 3, 3)
        self.w = (nn.Parameter(torch.randn(1, num_features)))

    def forward(self, x):
        return vmap(self.linear)(x)
        # out = (self.linear(x * torch.sin(self.w))**2).sum()
        # out.backward()
        # return list(self.parameters())


batch_size = 4
def f(x, w, b):
    out = torch.conv2d(x, w, b)
    return out
ws = [torch.nn.Conv2d(3, 6, 3).weight for _ in range(batch_size)]
ws = torch.stack(ws).detach()

bs = [torch.nn.Conv2d(3, 6, 3).bias for _ in range(batch_size)]
bs = torch.stack(bs).detach()

inps = (torch.randn(batch_size, 3, 3,10,10), ws, bs)

print(torch.conv2d(inps[0][0], inps[1][0], None).shape)
fx_graph = pythonkey_trace(wrap_key(vmap(f, in_dims=(0, 0, 0), out_dims=0), inps))
print(fx_graph(*inps).shape)
exit(0)

vmap_f = vmap(f, in_dims=(0,None))
vmap_f(*inps)
exit(0)
begin = time.time()
vmap_f(*inps)
print(time.time()-begin)
exit(0)

from functorch import vmap
batch_size, feature_size = 3, 5

def model(weights,feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss

torch._C._debug_only_display_vmap_fallback_warnings(True)
weights = torch.randn(feature_size, requires_grad=True)
examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)
inputs = (weights,examples, targets)
grad_weight_per_example = fx.symbolic_trace(key_wrap(vmap(grad(compute_loss, diff_argnums=(0,)), in_dims=(None, 0, 0)), inputs))
print(grad_weight_per_example.code)
# nnc_compile(grad_weight_per_example, inputs)
# print(grad_weight_per_example.code)


exit(0)
# grad_foo = ModuleBackward(f, inps)
# nnc_grad = nnc_compile(grad_foo, inps)

# Phabricate sample inputs
num_features = 128
def gen_inputs():
    x = torch.randn(4, num_features)
    inps = (x,)
    return inps
inps = gen_inputs()

iters = 10000
lr = 0.001

torch.manual_seed(3)
f = Foo(num_features)

train_mod = ModuleBackward(f, inps)
for p in f.parameters():
    p.grad = None
grad_f = fx.symbolic_trace(train_mod)
print(grad_f.code)
nnc_grad = nnc_compile(grad_f, inps)


avg = -1
eps = 0.01
begin = time.time()
for iter in range(iters):
    inps = gen_inputs()
    grads = nnc_grad(*inps)
    avg = (1-eps)*avg + eps*grads[0].sum()
    # print(grads)
    # if iter % 1000 == 0:
    #     print(f"itr {iter}: {avg}")
    for p, g in zip(grad_f.parameters(), grads):
        p.data -= lr*g
print("NNC train runtme: ", time.time()-begin)
print()

torch.manual_seed(3)
avg = -1
f = Foo(num_features)
begin = time.time()
for iter in range(iters):
    inps = gen_inputs()
    f(*inps).backward()
    grads = [p.grad for p in f.parameters()]
    avg = (1-eps)*avg + eps*grads[0].sum()
    # if iter % 1000 == 0:
    #     print(f"itr {iter}: {avg}")
    # print(grads)
    for p, g in zip(f.parameters(), grads):
        p.data -= lr*g
        p.grad = None
print("eager train runtime: ", time.time()-begin)
exit(0)



grad_f(*inps)

iters = 1000
# for _ in range(3):
#     grads = grad_f(*inps)

# if isinstance(grads, torch.Tensor):
#     print(grads.sum())
# else:
#     print([i.sum() for i in grads])
begin = time.time()
for _ in range(iters):
    grad_f(*inps)
print(time.time() - begin)


# for inp in inps:
#     inp.requires_grad = True

f(*inps).backward()
print([v.grad.sum() for v in f.parameters()])


begin = time.time()
for _ in range(iters):
    f(*inps).backward()
print(time.time() - begin)

# exit(0)

# def f(a, b):
#     return (a*b).sum()
# inps = (torch.randn(3000), torch.randn(3000))
# grad_f = fx.symbolic_trace(grad(f, inps))
# print(grad_f.code)
