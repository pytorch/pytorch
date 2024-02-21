import time
import timeit

import numpy as np

import torch


def add1(x):
    return x + 1


def bench(name, fn):
    x = torch.randn(1)
    start = time.perf_counter()
    for _ in range(3):
        fn(x)
    end = time.perf_counter()

    results = timeit.repeat(lambda: fn(x), number=1000, repeat=1000)
    print(f"{name} {np.median(results)*1000:.1f}us (warmup={end-start:.1f}s)")


def main():
    bench("eager   ", add1)
    bench("compiled", torch.compile(add1))


if __name__ == "__main__":
    main()
