import time
import torch
from torch._subclasses.fake_tensor import FakeTensorMode as PyFakeTensorMode


def to_fake(tensor):
    return torch._C._to_fake(tensor)


class CppFakeTensorMode:
    def __enter__(self):
        self._guard = torch._C._IncludeDispatchKeyGuard(
            torch._C.DispatchKey.Fake
        )
        self._guard.__enter__()
        return self

    def __exit__(self, *args):
        self._guard.__exit__(*args)


SIZES = [
    (16, 16, 16),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (128, 4096, 128),
    (4096, 128, 4096),
]

WARMUP = 3
ITERS = 100


def bench_cpu(m, k, n):
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    for _ in range(WARMUP):
        torch.mm(a, b)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        torch.mm(a, b)
    return (time.perf_counter() - t0) / ITERS


def bench_meta(m, k, n):
    a = torch.randn(m, k, device="meta")
    b = torch.randn(k, n, device="meta")
    for _ in range(WARMUP):
        torch.mm(a, b)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        torch.mm(a, b)
    return (time.perf_counter() - t0) / ITERS


def bench_cpp_fake(m, k, n):
    a = to_fake(torch.randn(m, k))
    b = to_fake(torch.randn(k, n))
    with CppFakeTensorMode():
        for _ in range(WARMUP):
            torch.mm(a, b)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            torch.mm(a, b)
        return (time.perf_counter() - t0) / ITERS


def bench_py_fake(m, k, n):
    with PyFakeTensorMode() as mode:
        a = mode.from_tensor(torch.randn(m, k))
        b = mode.from_tensor(torch.randn(k, n))
        for _ in range(WARMUP):
            torch.mm(a, b)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            torch.mm(a, b)
        return (time.perf_counter() - t0) / ITERS


def main():
    # C++ Fake goes through the boxed fallback -> our fast meta_mm kernel.
    # Comparing Meta vs C++ Fake shows the boxed fallback overhead.
    header = (
        f"{'Size':>20s}  {'CPU':>10s}  {'Meta':>10s}  "
        f"{'C++ Fake':>10s}  {'Py Fake':>10s}  "
        f"(all times in us)"
    )
    print(header)
    print("-" * len(header))

    for m, k, n in SIZES:
        label = f"{m}x{k}x{n}"
        t_cpu = bench_cpu(m, k, n)
        t_meta = bench_meta(m, k, n)
        t_cpp = bench_cpp_fake(m, k, n)
        t_py = bench_py_fake(m, k, n)
        print(
            f"{label:>20s}  {t_cpu*1e6:10.1f}  {t_meta*1e6:10.1f}  "
            f"{t_cpp*1e6:10.1f}  {t_py*1e6:10.1f}"
        )


if __name__ == "__main__":
    main()
