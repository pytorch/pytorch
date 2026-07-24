import sys

from benchmark_base import BenchmarkBase

import torch


class _DisableOverheadBase(BenchmarkBase):
    # Base for torch._dynamo.disable per-call overhead microbenchmarks: a
    # subclass sets _CATEGORY/_DESCRIPTION and _setup() to assign self._fn.
    def __init__(self):
        super().__init__(
            category=self._CATEGORY,
            device="cpu",
        )

    def name(self):
        return self.category()

    def description(self):
        return self._DESCRIPTION

    def _prepare_once(self):
        torch._dynamo.reset()
        self._setup()
        self.a = torch.ones(1)

        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._fn(self.a)


class Benchmark(_DisableOverheadBase):
    # Per-call overhead of the disable wrapper (toggling the eval-frame handler),
    # measured with a trivial callee so the count reflects the wrapper.
    _CATEGORY = "disable_overhead"
    _DESCRIPTION = "per-call overhead of torch._dynamo.disable with a trivial callee"

    def _setup(self):
        def f(x):
            return x

        self._fn = torch._dynamo.disable(f)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
