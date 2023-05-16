from functools import partial
import numpy as np
import pandas as pd
import timeit
import torch
from functorch.compile import pointwise_operator

WRITE_CSV = False
CUDA = False
SIZES = [1, 512, 8192]
NUMBER = [100, 10, 1, 1]
REPEAT = 20


@pointwise_operator
def nnc_add(a, b):
    return a + b


@pointwise_operator
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


def maybe_synced(fn):
    if CUDA:
        synchronize = torch.cuda.synchronize
        synchronize()  # warmup

        def _fn():
            result = fn()
            synchronize()
            return result

        return _fn
    return fn


def benchmark_loop(setup):
    result = np.zeros((REPEAT, len(SIZES), 2), dtype=np.float64)
    for s, n in enumerate(SIZES):
        nnc, aten = setup(n)
        nnc = maybe_synced(nnc)
        aten = maybe_synced(aten)

        for r in range(result.shape[0]):
            result[r, s, 0] = timeit.timeit(nnc, number=NUMBER[s])
            result[r, s, 1] = timeit.timeit(aten, number=NUMBER[s])

    result = np.median(result, axis=0)
    assert result.shape == (len(SIZES), 2)
    result = result[:, 1] / result[:, 0]
    print(result)
    return result


def test(make_args, nnc=nnc_add, aten=torch.add):
    def setup(n):
        args = make_args(n)
        result_aten = aten(*args)
        result_nnc = nnc(*args)
        assert result_nnc.dtype == result_aten.dtype
        assert result_nnc.size() == result_aten.size()
        assert result_nnc.stride() == result_aten.stride()
        torch.testing.assert_close(result_aten, result_nnc)
        return (lambda: nnc(*args), lambda: aten(*args))

    return benchmark_loop(setup)


def test_inplace(make_args, nnc=nnc_add, aten=torch.add):
    def inplace_setup(n):
        a, b = make_args(n)
        result_aten = torch.clone(a)
        result_nnc = torch.clone(a)
        nnc(result_nnc, b, out=result_nnc)
        aten(result_aten, b, out=result_aten)
        torch.testing.assert_close(result_aten, result_nnc)
        return (lambda: nnc(a, b, out=a), lambda: aten(a, b, out=a))

    return benchmark_loop(inplace_setup)


def test_out(make_args, out, nnc=nnc_add, aten=torch.add):
    def out_setup(n):
        args = make_args(n)
        result_aten = out(n)
        result_nnc = out(n)
        aten(*args, out=result_aten)
        nnc(*args, out=result_nnc)
        torch.testing.assert_close(result_aten, result_nnc)
        result = out(n)
        return (lambda: nnc(*args, out=result), lambda: aten(*args, out=result))

    return benchmark_loop(out_setup)


def test_backwards(make_args, nnc=nnc_add, aten=torch.add):
    def backwards_setup(n):
        args = make_args(n)
        (grad_var,) = (a for a in args if a.requires_grad)
        aten(*args).sum().backward()
        correct = grad_var.grad.clone()
        grad_var.grad.zero_()
        nnc(*args).sum().backward()
        torch.testing.assert_close(correct, grad_var.grad)
        return (
            lambda: nnc(*args).sum().backward(),
            lambda: aten(*args).sum().backward(),
        )

    return benchmark_loop(backwards_setup)


def main():
    torch.set_num_threads(1)  # TODO(jansel): add parallel support
    torch._C._jit_override_can_fuse_on_cpu(True)

    device = "cuda" if CUDA else "cpu"
    I = partial(torch.randint, 0, 100, device=device)
    R = partial(torch.randn, device=device)

    results = [
        ("add", test(lambda n: (R(n, n), R(n, n)))),
        ("broadcast1", test(lambda n: (R(n, n), R(1)))),
        ("broadcast2", test(lambda n: (R(n, n), R(n, 1)))),
        ("broadcast3", test(lambda n: (R(n, 1), R(1, n)))),
        ("inplace", test_inplace(lambda n: (R(n, n), R(n, 1)))),
        ("out=", test_out(lambda n: (R(n, n), R(n, n)), out=lambda n: R(n, n))),
        ("transposed1", test(lambda n: (R(n, n), R(n, n).transpose(0, 1)))),
        (
            "transposed2",
            test(lambda n: (R(n, n).transpose(0, 1), R(n, n).transpose(0, 1))),
        ),
        ("slice1", test(lambda n: (R(n + 1, n + 1, 2)[:n, :n, 0], R(n, n)))),
        ("slice2", test(lambda n: (R(n, n, 2)[:, :, 0], R(n, n, 2)[:, :, 0]))),
        (
            "strided out",
            test_out(
                lambda n: (R(n, n), R(n, n)),
                out=lambda n: R(n + 1, n + 1, 2)[:n, :n, 0],
            ),
        ),
        (
            "out convert",
            test_out(
                lambda n: (R(n, n), R(n, n)), out=lambda n: R(n, n, dtype=torch.float64)
            ),
        ),
        ("issue #57611 (n,32,32,2)", test(lambda n: (R(1, 32, 32, 2), R(n, 1, 1, 2)))),
        ("float+double", test(lambda n: (R(n, n), R(n, n, dtype=torch.float64)))),
        (
            "int+long",
            test(
                lambda n: (I([n, n], dtype=torch.int32), I([n, n], dtype=torch.int64))
            ),
        ),
        (
            "int+short",
            test(
                lambda n: (I([n, n], dtype=torch.int32), I([n, n], dtype=torch.int16))
            ),
        ),
        (
            "float+int",
            test(
                lambda n: (R([n, n], dtype=torch.float32), I([n, n], dtype=torch.int32))
            ),
        ),
        (
            "double+long",
            test(
                lambda n: (R([n, n], dtype=torch.float64), I([n, n], dtype=torch.int64))
            ),
        ),
        (
            "fused addnorm",
            test(
                lambda n: (R(n, n), R(n, n), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=eager_addnorm,
            ),
        ),
        (
            "fused addnorm (vs TS)",
            test(
                lambda n: (R(n, n), R(n, n), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=ts_addnorm,
            ),
        ),
        (
            "fused addnorm out=",
            test_out(
                lambda n: (R(n, n), R(n, n), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=inplace_addnorm,
                out=lambda n: R(n, n),
            ),
        ),
        (
            "fused addnorm out= (vs TS)",
            test_out(
                lambda n: (R(n, n), R(n, n), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=ts_ip_addnorm,
                out=lambda n: R(n, n),
            ),
        ),
        (
            "fused addnorm backward",
            test_backwards(
                lambda n: (R(n, n), R(n, n, requires_grad=True), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=eager_addnorm,
            ),
        ),
        (
            "fused addnorm backward (vs TS)",
            test_backwards(
                lambda n: (R(n, n), R(n, n, requires_grad=True), R(n, n), R(n, n)),
                nnc=nnc_addnorm,
                aten=ts_addnorm,
            ),
        ),
    ]

    df = pd.DataFrame(
        np.stack([r for n, r in results]),
        columns=[f"{n}x{n}".rjust(9) for n in SIZES],
        index=[n for n, r in results],
    )

    if WRITE_CSV:
        df.to_csv("../operator_authoring_results.csv")
        print("wrote ../operator_authoring_results.csv")

    print()
    print("Speedups over aten")
    pd.options.display.float_format = "{:.2f}x".format
    print(df)


if __name__ == "__main__":
    main()
