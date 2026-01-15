import sys
import time

import torch


DEPTH = 100


def gn(x):
    for _ in range(DEPTH):
        x = x + 1
    return x


def make_fn(next_fn):
    if next_fn is None:

        def fn(x):
            x = gn(x)
            torch._dynamo.graph_break()
            return gn(x)
    else:

        def fn(x):
            return gn(next_fn(gn(x)))

    # to prevent recompilation + fallback to eager
    fn.__code__ = fn.__code__.replace()
    return fn


fns = [make_fn(None)]
for _ in range(DEPTH):
    fns.append(make_fn(fns[-1]))

top_fn = fns[-1]

sys.setrecursionlimit(100000)
torch._dynamo.set_recursion_limit(1000000)


def main():
    start = time.perf_counter()
    print(top_fn(torch.ones(3)))
    end = time.perf_counter()

    print(f"eager total time: {end - start:.2f}s")

    opt_fn = torch.compile(top_fn, backend="eager")

    torch._dynamo.config.nested_graph_breaks = True
    start = time.perf_counter()
    opt_fn(torch.ones(3))
    end = time.perf_counter()

    print(f"nested_graph_breaks=True total time: {end - start:.2f}s")

    torch.compiler.reset()

    torch._dynamo.config.nested_graph_breaks = False
    start = time.perf_counter()
    opt_fn(torch.ones(3))
    end = time.perf_counter()

    print(f"nested_graph_breaks=False total time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
