import inspect
import itertools
import sys
import time

import click

import torch


torch.set_num_threads(1)
torch._C._debug_set_fusion_group_inlining(False)


def rand(*shape):
    return torch.rand(*shape).mul(16).add(1)


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------
def scalar():
    return (rand(1), rand(1))


def small():
    return (rand(32), rand(32))


def small_2d():
    return (rand(1, 32), rand(1, 32))


def small_broadcast():
    return (rand(4, 32), rand(32))


def medium():
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))


def medium_sliced():
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])


def medium_transpose():
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )


def medium2():
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))


def medium3d():
    return (rand(16, 32, 64), rand(16, 32, 64))


def medium_channels_last():
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
    )


def medium_broadcast():
    return (rand(32, 12, 64, 64), rand(64))


def medium_broadcast_channels_last():
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))


def large():
    return (rand(8192, 8192), rand(8192, 8192))


def large_transpose():
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))


def large_channels_last():
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
    )


def broadcast_narrow_57611():
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))


def large_broadcast_66816():
    return (rand(64, 8, 256, 162), rand(256, 162))


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------
def add(a, b):
    return 3 * a + b


def sub(a, b):
    return 3 * a - b


def mul(a, b):
    return 3 * a * b


def div(a, b):
    return 3 * a / b


def relu(a):
    return (3 * a).relu()


def sigmoid(a):
    return (3 * a).sigmoid()


def tanh(a):
    return (3 * a).tanh()


def log(a):
    return (3 * a).log()


def exp(a):
    return (3 * a).exp()


def square(a):
    return (3 * a) ** 2


def fma(a, b):
    return a * b + b


def mul_mul_add_66816(a, b, c):
    return (a * b) + (a * c)


def hardswish_int(a):
    return a * (a + 3).clamp(0, 6) / 6


def hardswish(a):
    return a * (a + 3).clamp(0.0, 6.0) / 6


def native_hardswish(a):
    return torch._C._nn.hardswish(a * 3)


def softplus(a):
    return (a * 1.0).exp().log1p() / 1.0


def mish(a):
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()


SHAPES = [
    scalar,
    small,
    small_2d,
    small_broadcast,
    medium,
    medium2,
    medium3d,
    medium_sliced,
    medium_transpose,
    medium_channels_last,
    medium_broadcast,
    medium_broadcast_channels_last,
    large,
    large_transpose,
    large_channels_last,
    broadcast_narrow_57611,
    large_broadcast_66816,
]

OPERATORS = [
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    tanh,
    log,
    exp,
    square,
    fma,
    mul_mul_add_66816,
    hardswish_int,
    hardswish,
    native_hardswish,
    softplus,
    mish,
]


def time_cpu(fn, args, iters):
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s


def time_cuda(fn, args, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    timer(fn, args, 3)
    calibration = timer(fn, args, 1)
    iters = int(1.0 / calibration)
    return timer(fn, args, iters) / iters


def benchmark(fn, args):
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    return f"{s * 1e6:.1f}"


def with_nvfuser():
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_nnc():
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def with_legacy():
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)


@click.command()
@click.option("--operators", default=None)
@click.option("--shapes", default=None)
def run_benchmarks(operators, shapes):
    if operators is None:
        operators = OPERATORS
    else:
        operators = [globals()[k] for k in operators.split(",")]
    if shapes is None:
        shapes = SHAPES
    else:
        shapes = [globals()[k] for k in shapes.split(",")]

    print("fuser,device,operator,shape,time")
    results = []
    for shape, operator in itertools.product(shapes, operators):
        nargs = len(inspect.signature(operator).parameters)
        args = shape()
        if nargs > len(args):
            args = list(args)
            args += [args[-1]] * (nargs - len(args))
        args = args[:nargs]
        args = [arg.to("cuda") for arg in args]

        result = benchmark(operator, args)
        print(
            ",".join(
                [
                    "eager",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(result),
                ]
            )
        )

        def bench(name):
            nnc_op = torch.jit.trace(operator, args)
            result = benchmark(nnc_op, args)
            print(
                ",".join(
                    [
                        name,
                        args[0].device.type,
                        operator.__name__,
                        shape.__name__,
                        micros(result),
                    ]
                )
            )
            sys.stdout.flush()

        with_nnc()
        bench("nnc")
        with_nvfuser()
        bench("nvfuser")
        with_legacy()
        bench("legacy")


if __name__ == "__main__":
    run_benchmarks()
