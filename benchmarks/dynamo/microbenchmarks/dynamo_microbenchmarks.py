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


def main():
    def fn():
        torch._dynamo.reset()
        symbolic_convert_overhead_stress_test(x, y, 100000)

    x = torch.randn(16)
    y = torch.randn(16)
    t = min(timeit.repeat(fn, number=1, repeat=3))
    print(f"symbolic_convert_overhead_stress_test: {t:.1f}s")


def profile():
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


if __name__ == "__main__":
    main()
    profile()
