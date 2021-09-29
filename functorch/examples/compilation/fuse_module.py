from functorch import compiled_module, tvm_compile
import torch.nn as nn
import torch
from functools import partial

def nop(f, _):
    return f

fw_compiler = partial(tvm_compile, name='fw_keops')
bw_compiler = partial(tvm_compile, name='bw_keops')
fw_compiler = nop
bw_compiler = nop

def run(mod, input):
    out = mod(input)
    out.sum().backward()
    grads = [p.grad for p in mod.parameters()]
    return (out, *grads)

class Foo(nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (self.param * x).sum(dim=0)

input = torch.randn(1)
mod = Foo()
compiled_mod = compiled_module(mod, fw_compiler, bw_compiler)

print("initial param: ", mod.param)
for a, b in zip(run(mod, input), run(compiled_mod, input)):
    print(a)
    print(b)
    torch.testing.assert_allclose(a, b)

out = mod(input)
out.sum().backward()
mod.param.data -= mod.param.grad
compiled_mod.orig_module.param.data -= compiled_mod.orig_module.param.grad
compiled_mod.orig_module.param.grad = None

print("orig param: ", mod.param.data)
print("compiled param: ", compiled_mod.orig_module.param.data)

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    print(a)
    print(b)
    torch.testing.assert_allclose(a, b)

import timeit
for _ in range(5):
    i = 10000
    t = timeit.Timer("mod(input)", globals=globals()).timeit(10000)
    print(f"eager {t/i*1e6}")
    t = timeit.Timer("compiled_mod(input)", globals=globals()).timeit(10000)
    print(f"compiled {t/i*1e6}")
