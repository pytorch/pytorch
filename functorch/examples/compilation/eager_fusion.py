from functorch import compiled_function, tvm_compile
import torch.nn as nn
import torch
from functools import partial
import time

a = torch.randn(10000, 1, 4, requires_grad=True)
b = torch.randn(1, 10000, 4)
def f(a):
    return (a * b).sin().sum(dim=0)

fw_compiler = partial(tvm_compile, name='fw_keops')
bw_compiler = partial(tvm_compile, name='bw_keops')
# fw_compiler = lambda x, _: x
# bw_compiler = lambda x, _: x
compiled_f = compiled_function(f, fw_compiler, bw_compiler).apply

iters = 10
out = compiled_f(a)
out.sum().backward()
def bench(func):
    begin = time.time()
    for _ in range(iters):
        out = func(a)
        out.sum().backward()
        mod.zero_grad()
    print(time.time()-begin)

def bench_jax():
    import jax.numpy as jnp
    import jax
    jax_a = jnp.array(a.detach().numpy())
    jax_b = jnp.array(b.detach().numpy())
    def f(a):
        return jnp.sin((a*jax_b).sum(axis=[0])).sum()
    jit_f = jax.jit(jax.grad(f))
    jit_f(jax_a)
    begin = time.time()
    for _ in range(iters):
        out = jit_f(jax_a)
    out.block_until_ready()
    print(time.time()-begin)
    # for

bench(f)
bench(compiled_f)
# bench_jax()
