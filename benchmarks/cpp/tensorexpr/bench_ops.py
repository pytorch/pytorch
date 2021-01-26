import timeit
import torch

torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._debug_set_fusion_group_inlining(False)
torch.set_num_threads(1)


def hardswish(x):
    return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


unary_ops = [
    hardswish,
    torch._C._nn.hardswish,
    torch.sigmoid,
    torch.reciprocal,
    torch.neg,
    torch.relu,
    torch.isnan,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.exp,
    torch.expm1,
    torch.erf,
    torch.erfc,
    torch.cos,
    torch.sin,
    torch.tan,
    torch.acos,
    torch.asin,
    torch.cosh,
    torch.sinh,
    torch.atan,
    torch.tanh,
    torch.sqrt,
    torch.rsqrt,
    torch.abs,
    torch.ceil,
    torch.floor,
    torch.round,
    torch.trunc,
    torch.lgamma,
]

print("{:20s} {:>10s} {:>10s} {:>10s}".format("op", "eager", "nnc", "speedup"))

for op in unary_ops:
    x = torch.rand((1024, 1024))
    traced = torch.jit.trace(lambda x: op(x), (x))

    # Warmup.
    warmup_iters = 8
    for _ in range(warmup_iters):
        op(x)
        traced(x)

    # Validate result.
    torch.testing.assert_allclose(op(x), traced(x))

    # Benchmark.
    bench_iters = 100
    teager = timeit.timeit(stmt="op(x)", globals=globals(), number=bench_iters)
    tjit = timeit.timeit(stmt="traced(x)", globals=globals(), number=bench_iters)
    print(f"{op.__name__:20s} {teager:10.3f} {tjit:10.3f} {teager/tjit:10.2f}")
