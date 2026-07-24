import time
import timeit

import numpy as np

import torch


def add1(x):
    return x + 1


def bench(name, fn, requires_grad):
    torch._dynamo.reset()
    x = torch.randn(1, requires_grad=requires_grad)
    start = time.perf_counter()
    for _ in range(3):
        fn(x)
    end = time.perf_counter()

    results = timeit.repeat(lambda: fn(x), number=1000, repeat=1000)
    median_us = np.median(results) * 1000
    print(f"{name} {median_us:.1f}us (warmup={end - start:.1f}s)")
    return median_us


def bench_overhead(label, requires_grad):
    print(label)
    eager_us = bench("eager   ", add1, requires_grad)
    compiled_us = bench("compiled", torch.compile(add1), requires_grad)
    # Steady-state per-call cost torch.compile adds on top of eager for a cache
    # hit: eval-frame interception, cache lookup, guard check, and the
    # compile_wrapper prologue/epilogue that saves and restores global state.
    print(f"overhead {compiled_us - eager_us:.1f}us")
    print()


def main():
    bench_overhead("requires_grad=False", False)
    bench_overhead("requires_grad=True", True)
    with torch.inference_mode():
        bench_overhead("inference_mode()", False)

    # torch._dynamo.disable's wrapper just toggles the eval-frame handler.
    print("torch._dynamo.disable")
    eager_us = bench("eager   ", add1, False)
    disabled_us = bench("disabled", torch._dynamo.disable(add1), False)
    print(f"overhead {disabled_us - eager_us:.1f}us")


if __name__ == "__main__":
    main()
