import torch.jit.te
import numpy as np
import pandas as pd
import math
from torch import randn
import timeit

torch.set_num_threads(1)  # TODO(jansel): add parallel support

SIZES = [2 ** n for n in range(0, 13, 4)]
NUMBER = [1000, 100, 10, 1]
REPEAT = 10

@torch.jit.te.pointwise_operator
def nnc_add(a, b):
    return a + b


@torch.jit.te.pointwise_operator
def nnc_norm(v, mean, std):
    return (v - mean) / std


def make_setup(make_args, nnc=nnc_add, aten=torch.add, inplace=False):
    def inplace_setup(n):
        a, b = make_args(n)
        result_aten = torch.clone(a)
        result_nnc = torch.clone(a)
        nnc(result_nnc, b, out=result_nnc)
        aten(result_aten, b, out=result_aten)
        torch.testing.assert_allclose(result_aten, result_nnc)
        return (lambda: nnc(a, b, out=a),
                lambda: aten(a, b, out=a))

    def setup(n):
        args = make_args(n)
        result_aten = aten(*args)
        result_nnc = torch.randn_like(result_aten)
        nnc(*args, out=result_nnc)
        torch.testing.assert_allclose(result_aten, result_nnc)
        result = torch.empty_like(result_aten)
        return (lambda: nnc(*args, out=result),
                lambda: aten(*args, out=result))

    if inplace:
        return inplace_setup
    else:
        return setup


def benchmark_loop(setup):
    result = np.zeros((REPEAT, len(SIZES), 2), dtype=np.float64)
    for s, n in enumerate(SIZES):
        nnc, aten = setup(n)

        for r in range(result.shape[0]):
            result[r, s, 0] = timeit.timeit(nnc, number=NUMBER[s])
            result[r, s, 1] = timeit.timeit(aten, number=NUMBER[s])

    result = np.median(result, axis=0)
    assert result.shape == (len(SIZES), 2)
    result = result[:, 1] / result[:, 0]
    print(result)
    return result


def benchmark(*args, **kwargs):
    return benchmark_loop(make_setup(*args, **kwargs))


def main():
    results = [
        ("(n,n)+(1)", benchmark(lambda n: (randn(n, n), randn(1)))),
        ("(n,n)+=(1)", benchmark(lambda n: (randn(n, n), randn(1)), inplace=True)),
        ("(n,n)+(n,1)", benchmark(lambda n: (randn(n, n), randn(n, 1)))),
        ("(n,n)+=(n,1)", benchmark(lambda n: (randn(n, n), randn(n, 1)), inplace=True)),
        ("(n,n)+(n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)))),
        ("(n,n)+=(n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)), inplace=True)),
        ("(n,1)+(1,n)", benchmark(lambda n: (randn(n, 1), randn(1, n)))),
        ("float+double", benchmark(lambda n: (randn(n, n), randn(n, n, dtype=torch.float64)))),
        ("issue 57611", benchmark(lambda n: (randn(1, 32, 32, 2), randn(n, 1, 1, 2)))),
    ]
    # TODO(jansel): implement int support
    print()
    print("Speedups over aten")
    print(pd.DataFrame(np.stack([r for n, r in results]),
                       columns=[f"2**{int(math.log(n, 2))}" for n in SIZES],
                       index=[n for n, r in results]).round(3))


if __name__ == '__main__':
    main()
    # main(4096)
