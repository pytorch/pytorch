import timeit

import torch.fx

N = 100000
K = 1000


def huge_graph():
    def fn(x):
        for _ in range(N):
            x = x.sin()
        return x

    return torch.fx.symbolic_trace(fn)


def main():
    g = huge_graph()

    def fn():
        for n in g.graph.nodes:
            pass

    t = min(timeit.repeat(fn, number=K, repeat=3))
    print(f"iterating over {N*K} FX nodes took {t:.1f}s ({N*K/t:.0f} nodes/s)")


if __name__ == "__main__":
    main()
