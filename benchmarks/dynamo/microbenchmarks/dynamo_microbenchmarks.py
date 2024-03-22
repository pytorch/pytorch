import cProfile
import pstats
import timeit

import torch


@torch.compile(backend="eager", fullgraph=True)
def symbolic_convert_overhead_stress_test(x, y, n):
    while n > 0:
        n -= 1
        x, y = y, x
    return x + y


@torch.compile(backend="eager", fullgraph=True)
def tensor_dicts(inputs):
    result = torch.zeros_like(inputs[0])
    for k1 in inputs:
        for k2 in inputs:
            result = result + torch.sin(inputs[k1] + inputs[k2])
    return result


def main1():
    def fn():
        torch._dynamo.reset()
        symbolic_convert_overhead_stress_test(x, y, 100000)

    x = torch.randn(16)
    y = torch.randn(16)
    t = min(timeit.repeat(fn, number=1, repeat=3))
    print(f"symbolic_convert_overhead_stress_test: {t:.1f}s")


def main2():
    def fn():
        torch._dynamo.reset()
        tensor_dicts(inputs)

    inputs = {i: torch.randn(1) for i in range(100)}
    t = min(timeit.repeat(fn, number=1, repeat=3))
    print(f"tensor_dicts: {t:.1f}s")


def profile1():
    x = torch.randn(16)
    y = torch.randn(16)
    torch._dynamo.reset()
    pr = cProfile.Profile()
    pr.enable()
    # 100k > 33k roughly cancels out the overhead of cProfile
    symbolic_convert_overhead_stress_test(x, y, 33000)
    pr.disable()
    ps = pstats.Stats(pr)
    ps.dump_stats("dynamo_microbenchmarks.prof")
    print("snakeviz dynamo_microbenchmarks.prof")


def profile2():
    inputs = {i: torch.randn(1) for i in range(100)}
    torch._dynamo.reset()
    pr = cProfile.Profile()
    pr.enable()
    tensor_dicts(inputs)
    pr.disable()
    ps = pstats.Stats(pr)
    ps.dump_stats("dynamo_microbenchmarks.prof")
    print("snakeviz dynamo_microbenchmarks.prof")


if __name__ == "__main__":
    main2()
    profile2()
