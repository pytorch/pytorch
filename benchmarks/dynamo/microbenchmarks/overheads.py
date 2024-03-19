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
    print(f"{name} {np.median(results)*1000:.1f}us (warmup={end-start:.1f}s)")


def main():
    print("requires_grad=False")
    bench("eager   ", add1, False)
    bench("compiled", torch.compile(add1), False)
    print()
    print("requires_grad=True")
    bench("eager   ", add1, True)
    bench("compiled", torch.compile(add1), True)
    print()
    print("inference_mode()")
    with torch.inference_mode():
        bench("eager   ", add1, False)
        bench("compiled", torch.compile(add1), False)


if __name__ == "__main__":
    main()
