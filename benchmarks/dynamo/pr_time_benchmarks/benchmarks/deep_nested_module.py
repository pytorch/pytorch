import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_cache


class DeepNestedModule(nn.Module):
    """A deeply nested module chain (depth-wise, not width-wise).

    This tests the performance of tracing deeply nested nn.Module hierarchies,
    which exercises AttrSource creation with long dotted member paths like
    "child.child.child.linear.weight".
    """

    def __init__(self, depth=40):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.child = DeepNestedModule(depth - 1)
            self.linear = nn.Linear(10, 10)
        else:
            self.linear = nn.Linear(10, 10)

    def forward(self, x):
        if self.depth > 0:
            x = self.child(x)
        return self.linear(x)


class Benchmark(BenchmarkBase):
    def __init__(
        self,
        ModuleClass,
        depth=40,
        backend="eager",
        is_gpu=False,
        dynamic=False,
    ):
        self.ModuleClass = ModuleClass
        self.depth = depth
        self._name = f"{ModuleClass.__name__}_depth{depth}"
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
        self.m = self.ModuleClass(depth=self.depth)
        torch.set_float32_matmul_precision("high")
        self.input = torch.ones(1, 10, device=self.device())

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        with fresh_cache():
            opt_m = torch.compile(backend=self.backend(), dynamic=self.is_dynamic())(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(DeepNestedModule, depth=40),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
