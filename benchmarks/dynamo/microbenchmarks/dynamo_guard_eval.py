import time
import timeit

import numpy as np

import torch
import torch._dynamo.config


# to satisfy linter complaining about undefined variable
foo = None

args = [f"x{i}" for i in range(100)]
fn_str = f"""\
def foo({", ".join(args)}):
    n = {" + ".join(arg + ".shape[0]" for arg in args)}
    return x0 + n
"""

exec(fn_str, globals())
torch._dynamo.config.cache_size_limit = 16


def bench(name, fn):
    torch._dynamo.reset()
    inps = [[torch.randn(i) for _ in range(100)] for i in range(10, 101, 10)]

    def run_fn():
        for inp in inps:
            fn(*inp)

    start = time.perf_counter()
    for _ in range(3):
        run_fn()
    end = time.perf_counter()

    results = timeit.repeat(lambda: run_fn(), number=1000, repeat=10)
    print(f"{name} {np.median(results) * 1000:.1f}us (warmup={end - start:.1f}s)")


def main():
    bench("compiled", torch.compile(foo, dynamic=False))  # type: ignore[F821]


if __name__ == "__main__":
    main()
