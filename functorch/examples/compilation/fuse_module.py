import timeit

import torch
import torch.nn as nn
from functorch.compile import compiled_module, tvm_compile


def nop(f, _):
    return f


fw_compiler = tvm_compile(target="llvm", tuning_logfile="fw_keops")
bw_compiler = tvm_compile(target="llvm", tuning_logfile="bw_keops")
fw_compiler = nop
bw_compiler = nop


def run(mod, input):
    out = mod(input)
    out.sum().backward()
    grads = [p.grad for p in mod.parameters()]
    return (out, *grads)


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.register_buffer("buf", torch.randn(1))

    def forward(self, x):
        return (self.param * x + self.buf).sum(dim=0)


input = torch.randn(1)
mod = Foo()
compiled_mod = compiled_module(mod, fw_compiler, bw_compiler)

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

out = mod(input)
out.sum().backward()
mod.param.data -= mod.param.grad
compiled_mod.orig_module.param.data -= compiled_mod.orig_module.param.grad
compiled_mod.orig_module.param.grad = None

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

for _ in range(5):
    i = 10000
    t = timeit.Timer("mod(input)", globals=globals()).timeit(10000)
    print(f"eager {t/i*1e6}")
    t = timeit.Timer("compiled_mod(input)", globals=globals()).timeit(10000)
    print(f"compiled {t/i*1e6}")
