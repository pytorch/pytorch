import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_inductor_cache


class ListOfLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Benchmark(BenchmarkBase):
    def __init__(self, ModuleClass, backend, is_gpu=False, dynamic=False):
        self.ModuleClass = ModuleClass
        self.backend = backend
        self._name = ModuleClass.__name__
        self._is_gpu = is_gpu
        self._dynamic = dynamic

    def name(self):
        prefix = f"basic_modules_{self._name}_{self.backend}"
        if self._dynamic:
            prefix += "_dynamic"
        if self._is_gpu:
            prefix += "_gpu"
        return prefix

    def _prepare_once(self):
        self.m = self.ModuleClass()
        torch.set_float32_matmul_precision("high")
        self.input = torch.ones(10, device="cuda" if self._is_gpu else "cpu")

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        with fresh_inductor_cache():
            opt_m = torch.compile(backend=self.backend, dynamic=self._dynamic)(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(ListOfLinears, "eager"),
        Benchmark(ListOfLinears, "inductor"),
        Benchmark(ListOfLinears, "inductor", is_gpu=True),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
