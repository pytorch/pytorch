import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__(
            category="deep_nested_module",
            backend="eager",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}_{self.device()}"

    def _prepare_once(self):
        self.m = DeepNestedModule(depth=40)
        self.input = torch.ones(1, 10, device=self.device())

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend=self.backend())
        def f(inp):
            return self.m(inp)

        f(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
