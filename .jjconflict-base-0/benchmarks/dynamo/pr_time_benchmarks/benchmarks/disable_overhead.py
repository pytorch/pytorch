import sys

from benchmark_base import BenchmarkBase

import torch


def _trivial(x):
    return x


class Benchmark(BenchmarkBase):
    # Per-call overhead of a disable-style wrapper around a trivial callee, so
    # the instruction count reflects the wrapper, not the callee body. `wrap` is
    # the decorator under test, applied to _trivial.
    def __init__(self, category, description, wrap):
        self._description = description
        self._wrap = wrap
        super().__init__(category=category, device="cpu")

    def name(self):
        return self.category()

    def description(self):
        return self._description

    def _prepare_once(self):
        torch._dynamo.reset()
        self._fn = self._wrap(_trivial)
        self.a = torch.ones(1)
        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._fn(self.a)


def _benchmarks():
    return [
        Benchmark(
            "disable_overhead",
            "per-call overhead of torch._dynamo.disable with a trivial callee",
            torch._dynamo.disable,
        ),
    ]


def main():
    result_path = sys.argv[1]
    for benchmark in _benchmarks():
        benchmark.enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
