import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_inductor_cache


class NestedModule(nn.Module):
    def __init__(self, depth=3, width=4):
        super().__init__()
        self.depth = depth
        self.width = width

        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()

        sub_mods = []
        if depth > 0:
            for i in range(width):
                sub_mods.append(NestedModule(depth - 1, width))
        else:
            for i in range(width):
                sub_mods.append(nn.ReLU())
        self.sub_mods = nn.Sequential(*sub_mods)
        self.a = 2

    def forward(self, x):
        x = self.relu_a(x)
        x = x + self.sub_mods(x)
        return x + self.relu_b(x) + self.a


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
        Benchmark(NestedModule),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
