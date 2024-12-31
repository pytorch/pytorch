import timeit

import torch.fx
from torch._inductor.codecache import FxGraphHashDetails


N = 10000
K = 100


def huge_graph():
    def fn(x):
        for _ in range(N):
            x = x.sin()
        return x

    return torch.fx.symbolic_trace(fn)


def main():
    g = huge_graph()
    details = FxGraphHashDetails(g, [], {}, [])

    def fn():
        return details.debug_lines()

    t = min(timeit.repeat(fn, number=K, repeat=3))
    print(f"iterating over {N*K} FX nodes took {t:.1f}s ({N*K/t:.0f} nodes/s)")


if __name__ == "__main__":
    main()
