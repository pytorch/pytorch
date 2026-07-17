import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # torch._dynamo.disable is used on hot paths (e.g. around custom operators),
    # so the per-call overhead of its wrapper -- toggling the eval-frame handler
    # off and back on around the call -- matters. Measure it with a trivial
    # callee so the count reflects the wrapper, not the callee body.
    def __init__(self):
        super().__init__(
            category="disable_overhead",
            device="cpu",
        )

    def name(self):
        return self.category()

    def description(self):
        return "per-call overhead of torch._dynamo.disable with a trivial callee"

    def _prepare_once(self):
        torch._dynamo.reset()

        def f(x):
            return x

        self._fn = torch._dynamo.disable(f)
        self.a = torch.ones(1)

        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._fn(self.a)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
