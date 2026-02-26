# Owner(s): ["module: mps"]
# Collection of op level benchmarks for MPS
# Useful as reference tool when migrating ops from MPS to Metal
import itertools
import sys
import timeit
import warnings
from typing import Optional

import torch
from torch.utils.benchmark import Compare, Measurement, Timer


def bench_unary_op(func, x, label) -> Measurement:
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x);{sync_cmd}",
        globals={"f": func, "x": x},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_binary_op(func, x, y, label) -> Measurement:
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x, y);{sync_cmd}",
        globals={"f": func, "x": x, "y": y},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)}, {str(y.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_unary(
    unary_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dtype)
    x_s = torch.testing.make_tensor(1024, 2048, device=device, dtype=dtype)[::, ::2]
    rc = []
    rc.append(bench_unary_op(unary_func, x, "dense"))
    rc.append(bench_unary_op(unary_func, x.t(), "transposed"))
    rc.append(bench_unary_op(unary_func, x_s, "strided"))
    rc.append(bench_unary_op(unary_func, x_s.t(), "strided + transposed"))
    return rc


def bench_binary(
    binary_func,
    device: str = "mps",
    dt_a: torch.dtype = torch.float32,
    dt_b: Optional[torch.dtype] = None,
) -> list[Measurement]:
    dt_b = dt_b if dt_b is not None else dt_a
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_a)
    y = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_b)
    s = torch.testing.make_tensor((), device=device, dtype=dt_b)
    rc = []
    rc.append(bench_binary_op(binary_func, x, y, "dense-dense"))
    rc.append(bench_binary_op(binary_func, x.t(), y.t(), "transp-transp"))
    rc.append(bench_binary_op(binary_func, x, y.t(), "dense-transp"))
    rc.append(bench_binary_op(binary_func, x.t(), y, "transp-dense"))
    rc.append(bench_binary_op(binary_func, x, s, "dense-scalar"))
    rc.append(bench_binary_op(binary_func, x, y[0], "dense-bcast"))
    return rc


def check_eager_vs_compile(rc_c, rc_e, func, dtype):
    if not torch.allclose(rc_c, rc_e):
        mdiff = (rc_c - rc_e).abs().max()
        warnings.warn(
            f"Eager and compile reduction do not match for {func.__name__} and {dtype} max_diff={mdiff}",
            stacklevel=2,
        )


def bench_reduction(
    reduction_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    rc = []

    # Bench 2D with reduction over dim=0
    def f(t):
        return reduction_func(t, dim=0)

    f.__name__ = reduction_func.__name__
    f_c = torch.compile(f, dynamic=False, fullgraph=True)

    for size in (512, 1024, 2048, 4096):
        x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
        rc_c, rc_e = f(x), f_c(x)
        rc_c, rc_e = (rc_c[0], rc_e[0]) if isinstance(rc_c, tuple) else (rc_c, rc_e)
        check_eager_vs_compile(rc_c, rc_e, reduction_func, dtype)
        rc.append(bench_unary_op(f, x, f"eager-{size}x{size}"))
        rc.append(bench_unary_op(f_c, x, f"compile-{size}x{size}"))
    return rc


def bench_scan(
    scan_func,
    device: str = "mps",
    dtype: torch.dtype = torch.float32,
    with_indices: bool = False,
) -> list[Measurement]:
    rc = []

    # Bench cumsum along different dimensions
    for dim in [0, 1]:

        def f(t):
            return scan_func(t, dim=dim)

        f_c = torch.compile(f, dynamic=False, fullgraph=True)

        for size in (32, 128, 512, 1024):
            f.__name__ = f"{scan_func.__name__}-dim{dim}-{size}x{size}"
            f_c.__name__ = f.__name__
            x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
            rc_c, rc_e = f(x), f_c(x)
            if with_indices:
                check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
                check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
            else:
                check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
            rc.append(bench_unary_op(f, x, "eager"))
            rc.append(bench_unary_op(f_c, x, "compile"))

    # Bench 1D cumsum for different sizes
    def f_1d(t):
        return scan_func(t, dim=0)

    f_1d_c = torch.compile(f_1d, dynamic=False, fullgraph=True)

    for size in (100, 10000, 1000000):
        f_1d.__name__ = f"{scan_func.__name__}-1d-{size}"
        f_1d_c.__name__ = f_1d.__name__
        x = torch.testing.make_tensor(size, device=device, dtype=dtype)
        rc_c, rc_e = f_1d(x), f_1d_c(x)
        if with_indices:
            check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
            check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
        else:
            check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
        rc.append(bench_unary_op(f_1d, x, "eager"))
        rc.append(bench_unary_op(f_1d_c, x, "compile"))

    return rc


def bench_grid_sampler_2d(
    device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    rc = []
    sync_cmd = "torch.mps.synchronize()" if device == "mps" else ""

    for mode in ("bilinear", "nearest", "bicubic"):
        for size in (64, 128, 256, 512):
            N, C = 4, 16
            inp = torch.randn(N, C, size, size, device=device, dtype=dtype)
            grid = torch.rand(N, size, size, 2, device=device, dtype=dtype) * 2 - 1

            def f(inp, grid, mode=mode):
                return torch.nn.functional.grid_sample(
                    inp, grid, mode=mode, padding_mode="zeros", align_corners=False
                )

            f_c = torch.compile(f, dynamic=False, fullgraph=True)

            rc_e, rc_c = f(inp, grid), f_c(inp, grid)
            if not torch.allclose(rc_e, rc_c, atol=1e-2, rtol=1e-2):
                mdiff = (rc_e - rc_c).abs().max()
                warnings.warn(
                    f"Eager and compile grid_sample do not match for {mode} {dtype} max_diff={mdiff}",
                    stacklevel=2,
                )

            sub_label = f"grid_sample-{mode}-{size}x{size} ({str(dtype)})"
            for label, fn in [("eager", f), ("compile", f_c)]:
                t = Timer(
                    stmt=f"f(inp, grid);{sync_cmd}",
                    globals={"f": fn, "inp": inp, "grid": grid},
                    language="python",
                    timer=timeit.default_timer,
                    sub_label=sub_label,
                    description=label,
                    env=torch.__version__,
                )
                rc.append(t.blocked_autorange())

    return rc


def main() -> None:
    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    def bench_index_ops():
        B = 11
        rc = []
        for dtype, N in itertools.product(
            [torch.int8, torch.float16, torch.float32], [50, 100, 500, 1000, 2000]
        ):
            x = torch.testing.make_tensor((B, N, N), device="mps", dtype=dtype)
            y = torch.randint(0, B, (3,))
            rc.append(bench_binary_op(torch.Tensor.__getitem__, x, y, f"{B}x{N}x{N}"))
        Compare(rc).print()

    def bench_unary_ops():
        rc = []
        for op, dtype in itertools.product([torch.sqrt, torch.sin], dtypes):
            rc.extend(bench_unary(op, dtype=dtype))
        Compare(rc).print()

    def bench_reduction_ops():
        rc = []
        for op in [torch.sum, torch.max]:
            rc.extend(bench_reduction(op))
        Compare(rc).print()

    def bench_scan_ops():
        rc = []
        for dtype in dtypes:
            rc.extend(bench_scan(torch.cumsum, dtype=dtype))
        Compare(rc).print()

    def bench_scan_with_indices_ops():
        rc = []
        for dtype in dtypes:
            rc.extend(bench_scan(torch.cummin, dtype=dtype, with_indices=True))
        Compare(rc).print()

    def bench_binary_ops():
        rc = []
        ops = [torch.fmax, torch.add]
        for op, dtype in itertools.product(ops, dtypes):
            rc.extend(bench_binary(op, dt_a=dtype))
            if dtype == torch.float32:
                rc.extend(bench_binary(op, dt_b=torch.float16))
        Compare(rc).print()

    def bench_grid_sampler_2d_ops():
        rc = []
        for dtype in dtypes:
            rc.extend(bench_grid_sampler_2d(dtype=dtype))
        Compare(rc).print()

    benchmarks = {
        "index": bench_index_ops,
        "unary": bench_unary_ops,
        "reduction": bench_reduction_ops,
        "scan": bench_scan_ops,
        "scan_with_indices": bench_scan_with_indices_ops,
        "binary": bench_binary_ops,
        "grid_sampler_2d": bench_grid_sampler_2d_ops,
    }

    selected = sys.argv[1:]
    if not selected:
        selected = list(benchmarks.keys())
    for name in selected:
        if name not in benchmarks:
            print(
                f"Unknown benchmark: {name}. Available: {', '.join(benchmarks.keys())}"
            )
            sys.exit(1)

    for name in selected:
        print(f"Running {name}...")
        benchmarks[name]()


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 2**16
    main()
