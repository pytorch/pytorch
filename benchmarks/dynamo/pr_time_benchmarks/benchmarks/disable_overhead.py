import sys

from benchmark_base import BenchmarkBase

import torch


class _DisableOverheadBase(BenchmarkBase):
    # Shared skeleton for the torch._dynamo.disable family of per-call overhead
    # microbenchmarks. A subclass sets _CATEGORY/_DESCRIPTION and implements
    # _setup() to assign the callable(s) it measures; _work() invokes the target
    # with a single tensor arg. Kept as separate subclasses (rather than one
    # parametrized instance) so each reports its own instruction count on the
    # pr_time dashboard.
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
    # torch._dynamo.disable is used on hot paths (e.g. around custom operators),
    # so the per-call overhead of its wrapper -- toggling the eval-frame handler
    # off and back on around the call -- matters. Measure it with a trivial
    # callee so the count reflects the wrapper, not the callee body.
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
