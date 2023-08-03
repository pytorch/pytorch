import timeit
import torch
import torch.nn.functional as F

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

print(f"{'op':20s} {'eager':>10s} {'nnc':>10s} {'speedup':>10s}")

for op in unary_ops:
    x = torch.rand((1024, 1024))
    traced = torch.jit.trace(lambda x: op(x), (x))

    # Warmup.
    warmup_iters = 8
    for _ in range(warmup_iters):
        op(x)
        traced(x)

    # Validate result.
    torch.testing.assert_close(op(x), traced(x))

    # Benchmark.
    bench_iters = 100
    teager = timeit.timeit(stmt="op(x)", globals=globals(), number=bench_iters)
    tjit = timeit.timeit(stmt="traced(x)", globals=globals(), number=bench_iters)
    print(f"{op.__name__:20s} {teager:10.3f} {tjit:10.3f} {teager/tjit:10.2f}")

def test_batch_norm():
    op = F.batch_norm
    print(f"{'op':20s} {'shape':20s} {'eager':>10s} {'nnc':>10s} {'speedup':>10s}")
    batch_norm_shapes = [
        [1, 64, 112, 112],
        [1, 256, 14, 14],
        [1, 128, 28, 28],
        [1, 64, 56, 56],
        [1, 512, 7, 7],
        [5, 64, 112, 112],
        [5, 256, 14, 14],
        [5, 128, 28, 28],
        [5, 64, 56, 56],
        [5, 512, 7, 7]]
    for n, c, h, w in batch_norm_shapes:
        x = torch.rand((n, c, h, w))
        y = torch.rand(c)
        z = torch.rand(c)
        traced = torch.jit.trace(lambda x, y, z: op(x, y, z), (x, y, z))

        # Warmup.
        warmup_iters = 8
        for _ in range(warmup_iters):
            op(x, y, z)
            traced(x, y, z)

        # Validate result.
        torch.testing.assert_close(op(x, y, z), traced(x, y, z))

        # Benchmark.
        bench_iters = 100
        teager = timeit.timeit(stmt="op(x, y, z)", globals=locals(), number=bench_iters)
        tjit = timeit.timeit(stmt="traced(x, y, z)", globals=locals(), number=bench_iters)
        print(f"{op.__name__:20s} ({n:>3d}, {c:>3d}, {h:>3d}, {w:>3d}) {teager:10.3f} {tjit:10.3f} {teager/tjit:10.2f}")

test_batch_norm()
