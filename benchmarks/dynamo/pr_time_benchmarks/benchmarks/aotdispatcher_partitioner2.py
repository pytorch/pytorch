import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(
            category="aotdispatcher_partitioner",
            backend="aot_eager_decomp_partition",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}_{self.device()}2"

    def description(self):
        return """
Partitioner benchmark with many parallel use chains.
See https://github.com/pytorch/pytorch/issues/145081"""

    def _prepare_once(self):
        self.x = torch.randn(4, 4, requires_grad=True)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(x):
            tmps = [x + i for i in range(16)]
            tmps = [x + tmp for tmp in tmps]
            for i in range(len(tmps) - 4):
                tmps[i] = tmps[i].sin().mul(tmps[i])
                tmps[i + 1] -= tmps[i]
                tmps[i + 2] -= tmps[i]
                tmps[i + 3] -= tmps[i]
            return sum(tmps)

        f(self.x)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
