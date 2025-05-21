import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_inductor_cache


# Create a chain of artificial nesting
def fn(x):
    return x + 1


def fn1(x):
    return fn(x)


def fn2(x):
    return fn1(x)


def fn3(x):
    return fn2(x)


def fn4(x):
    return fn3(x)


def fn5(x):
    return fn4(x)


def fn6(x):
    return fn5(x)


def fn7(x):
    return fn6(x)


def fn8(x):
    return fn7(x)


def fn9(x):
    return fn8(x)


class InlineMod(nn.Module):
    def __init__(self):
        super().__init__()
        self._n = 1000

    def forward(self, x):
        for _ in range(self._n):
            x = fn9(x)
        return x


class Benchmark(BenchmarkBase):
    def __init__(
        self,
        ModuleClass,
        backend="eager",
        is_gpu=False,
        dynamic=False,
    ):
        self.ModuleClass = ModuleClass
        self._name = ModuleClass.__name__
        self._is_gpu = is_gpu

        super().__init__(
            category="basic",
            backend=backend,
            device="cuda" if self._is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        prefix = f"{self.category()}_{self._name}_{self.backend()}"
        return prefix

    def _prepare_once(self):
        self.m = self.ModuleClass()
        torch.set_float32_matmul_precision("high")
        self.input = torch.ones(10, device=self.device())

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        # enable_cpp_symbolic_shape_guards has impact on this benchmark
        # Keep using False value for consistency.
        with (
            fresh_inductor_cache(),
        ):
            opt_m = torch.compile(backend=self.backend(), dynamic=self.is_dynamic())(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(InlineMod),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
