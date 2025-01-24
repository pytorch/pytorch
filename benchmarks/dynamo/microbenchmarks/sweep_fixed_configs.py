import argparse
import functools
import math
import operator
import sys

from triton.testing import do_bench

import torch


product = functools.partial(functools.reduce, operator.mul)


def nearest_log_2(x: int) -> int:
    return int(round(math.log2(x)))


def transpose_storage(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).contiguous().transpose(0, 1)


def benchmark(args, shape):
    def eager():
        vals = sum(inputs)
        total = vals.sum(-1, keepdim=True)
        for _ in range(args.depth - 1):
            vals = vals - total
            total = vals.sum(-1, keepdim=True)
        return total

    @torch.compile(
        fullgraph=True,
        options={
            "max_autotune": True,
            "max_autotune_pointwise": True,
            "coordinate_descent_tuning": True,
        },
    )
    def max_autotune():
        return eager()

    @torch.compile(
        fullgraph=True,
        options={
            "triton.multi_kernel": True,
            "max_autotune": True,
            "max_autotune_pointwise": True,
            "coordinate_descent_tuning": True,
        },
    )
    def multi_kernel():
        return eager()

    @torch.compile(
        fullgraph=True,
        options={
            "triton.cooperative_reductions": args.cooperative,
        },
    )
    def fixed():
        return eager()

    dtype = getattr(torch, args.dtype)
    inputs = (
        [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(args.inner)]
        + [
            transpose_storage(torch.zeros(shape, dtype=dtype, device="cuda"))
            for _ in range(args.outer)
        ]
        + [
            torch.zeros([1, *shape[1:]], dtype=dtype, device="cuda")
            for _ in range(args.inner_broadcast)
        ]
        + [
            torch.zeros([*shape[:-1], 1], dtype=dtype, device="cuda")
            for _ in range(args.outer_broadcast)
        ]
    )
    mb = product([dtype.itemsize, sum(x.numel() for x in inputs)]) / 1024**2
    msg = [f"{shape}:".ljust(16)]
    expected = eager()
    times = []

    for fn in (
        eager,
        max_autotune,
        # multi_kernel,
        fixed,
    ):
        torch.compiler.reset()
        torch.testing.assert_close(fn(), expected, rtol=1e-4, atol=1e-4)
        sec = do_bench(fn, return_mode="median")
        times.append(sec)
        if mb / sec > 9.5:
            msg.append(f"{fn.__name__}={mb / sec:3.0f}mb/s")
        else:
            msg.append(f"{fn.__name__}={mb / sec:3.1f}mb/s")

    print(" ".join(msg), f"{min(times[:-1]) / times[-1]:.2f}x")
    sys.stdout.flush()
    return mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--inner-broadcast", type=int, default=0)
    parser.add_argument("--outer-broadcast", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--min", type=int, default=2**12)
    parser.add_argument("--max", type=int, default=2**22)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--cooperative", action="store_true")
    args = parser.parse_args()
    for r in range(nearest_log_2(args.min), nearest_log_2(args.max) + 1):
        benchmark(args, (args.batch_size, 2**r + args.offset))


if __name__ == "__main__":
    main()
