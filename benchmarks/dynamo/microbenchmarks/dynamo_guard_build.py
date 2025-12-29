import sys
import time

import torch


class Foo:
    pass


obj = Foo()

DEPTH = 2000

attrs = [f"attr{i}" for i in range(DEPTH)]

for i, attr in enumerate(attrs):
    setattr(obj, attr, i)

lst = obj

for _ in range(DEPTH):
    lst = [lst]

sys.setrecursionlimit(100000)
torch._dynamo.set_recursion_limit(1000000)


@torch.compile(backend="eager")
def fn(x):
    unpacked = lst
    for _ in range(DEPTH):
        unpacked = unpacked[0]
    for i in range(DEPTH):
        x = x + getattr(unpacked, f"attr{i}")
    return x


def main():
    opt_fn = torch.compile(fn, backend="eager")

    start = time.perf_counter()
    opt_fn(torch.randn(3))
    end = time.perf_counter()

    print(f"total time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
