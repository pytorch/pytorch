import torch.jit.te
import numpy as np
import pandas as pd
from torch import randn
import timeit
import functools

torch.set_num_threads(1)  # TODO(jansel): add parallel support
torch._C._jit_override_can_fuse_on_cpu(True)

SIZES = [2 ** n for n in range(0, 13, 4)]
NUMBER = [1000, 100, 10, 1]
REPEAT = 10
randi = functools.partial(torch.randint, 0, 100)


@torch.jit.te.pointwise_operator
def nnc_add(a, b):
    return a + b


@torch.jit.te.pointwise_operator
def nnc_addnorm(a, b, mean, std):
    return (a + b - mean) / std


def eager_addnorm(a, b, mean, std):
    return (a + b - mean) / std


def inplace_addnorm(a, b, mean, std, out):
    out = torch.add(a, b, out=out)
    torch.sub(out, mean, out=out)
    torch.div(out, std, out=out)
    return out


ts_addnorm = torch.jit.script(eager_addnorm)
ts_ip_addnorm = torch.jit.script(inplace_addnorm)


def make_setup(make_args, nnc=nnc_add, aten=torch.add, inplace=False, out=False):
    def setup(n):
        args = make_args(n)
        result_aten = aten(*args)
        result_nnc = nnc(*args)
        assert result_nnc.dtype == result_aten.dtype
        assert result_nnc.size() == result_aten.size()
        assert result_nnc.stride() == result_aten.stride()
        torch.testing.assert_allclose(result_aten, result_nnc)
        return (lambda: nnc(*args),
                lambda: aten(*args))

    def out_setup(n):
        args = make_args(n)
        result_aten = out(n)
        result_nnc = out(n)
        aten(*args, out=result_aten)
        nnc(*args, out=result_nnc)
        torch.testing.assert_allclose(result_aten, result_nnc)
        result = out(n)
        return (lambda: nnc(*args, out=result),
                lambda: aten(*args, out=result))

    def inplace_setup(n):
        a, b = make_args(n)
        result_aten = torch.clone(a)
        result_nnc = torch.clone(a)
        nnc(result_nnc, b, out=result_nnc)
        aten(result_aten, b, out=result_aten)
        torch.testing.assert_allclose(result_aten, result_nnc)
        return (lambda: nnc(a, b, out=a),
                lambda: aten(a, b, out=a))

    if inplace:
        return inplace_setup
    elif out:
        return out_setup
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
        ("(n)+(n)", benchmark(lambda n: (randn(n), randn(n)))),
        ("(n,1)+(1,n)", benchmark(lambda n: (randn(n, 1), randn(1, n)))),
        ("(n,n)+(1)", benchmark(lambda n: (randn(n, n), randn(1)))),
        ("(n,n)+=(1)", benchmark(lambda n: (randn(n, n), randn(1)), inplace=True)),
        ("(n,n)+(n,1)", benchmark(lambda n: (randn(n, n), randn(n, 1)))),
        ("(n,n)+=(n,1)", benchmark(lambda n: (randn(n, n), randn(n, 1)), inplace=True)),
        ("(n,n)+(n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)))),
        ("(n,n)+=(n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)), inplace=True)),
        ("out= (n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)), out=lambda n: randn(n, n))),
        ("issue 57611 (n,32,32,2)", benchmark(lambda n: (randn(1, 32, 32, 2), randn(n, 1, 1, 2)))),
        ("transposed1 (n,n)", benchmark(lambda n: (randn(n, n),
                                                   randn(n, n).transpose(0, 1)))),
        ("transposed2 (n,n)", benchmark(lambda n: (randn(n, n).transpose(0, 1),
                                                   randn(n, n).transpose(0, 1)))),
        ("slice1 (n,n)", benchmark(lambda n: (randn(n + 10, n + 10, 32)[:n, :n, 0],
                                              randn(n, n)))),
        ("slice2 (n,n)", benchmark(lambda n: (randn(n, n, 32)[:, :, 0],
                                              randn(n, n, 32)[:, :, 0]))),
        ("strided out (n,n)", benchmark(lambda n: (randn(n, n), randn(n, n)),
                                        out=lambda n: randn(n + 8, n + 8, 2)[:n, :n, 0], )),
        ("out convert (n,n)", benchmark(lambda n: (randn(n, n),
                                                   randn(n, n)),
                                        out=lambda n: randn(n, n, dtype=torch.float64))),
        ("float+double (n,n)", benchmark(lambda n: (randn(n, n), randn(n, n, dtype=torch.float64)))),
        ("int+long (n,n)", benchmark(lambda n: (randi([n, n], dtype=torch.int32),
                                                randi([n, n], dtype=torch.int64)))),
        ("int+short (n,n)", benchmark(lambda n: (randi([n, n], dtype=torch.int32),
                                                 randi([n, n], dtype=torch.int16)))),
        ("float+int (n,n)", benchmark(lambda n: (randn([n, n], dtype=torch.float32),
                                                 randi([n, n], dtype=torch.int32)))),
        ("double+long (n,n)", benchmark(lambda n: (randn([n, n], dtype=torch.float64),
                                                   randi([n, n], dtype=torch.int64)))),
        ("fused addnorm (vs eager)", benchmark(lambda n: (randn(n, n), randn(n, n), randn(n, n), randn(n, n)),
                                               nnc=nnc_addnorm, aten=eager_addnorm)),
        ("fused addnorm (vs TS)", benchmark(lambda n: (randn(n, n), randn(n, n), randn(n, n), randn(n, n)),
                                            nnc=nnc_addnorm, aten=ts_addnorm)),
        ("fused addnorm (out=, eager)", benchmark(lambda n: (randn(n, n), randn(n, n), randn(n, n), randn(n, n)),
                                                  nnc=nnc_addnorm, aten=inplace_addnorm, out=lambda n: randn(n, n))),
        ("fused addnorm (out=, TS)", benchmark(lambda n: (randn(n, n), randn(n, n), randn(n, n), randn(n, n)),
                                               nnc=nnc_addnorm, aten=ts_ip_addnorm, out=lambda n: randn(n, n))),
    ]
    # TODO(jansel): implement int support
    print()
    print("Speedups over aten")
    pd.options.display.float_format = ' {:.2f}x'.format
    print(pd.DataFrame(np.stack([r for n, r in results]),
                       columns=[f"n={n}" for n in SIZES],
                       index=[n for n, r in results]))


if __name__ == '__main__':
    main()
