import random
import sys

from benchmarks.dynamo.pr_time_benchmarks.benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    N = 20

    def name(self):
        return "update_hint_regression"

    def description(self):
        return "information at https://github.com/pytorch/pytorch/pull/129893"

    def prepare_once(self):
        torch._dynamo.config.capture_scalar_outputs = True
        random.seed(42)
        self.splits = torch.randint(10, (self.N,))
        sz = self.splits.sum().item()
        self.input = torch.randn(sz)

    def prepare(self):
        torch._dynamo.reset()

    def work(self):
        @torch.compile(fullgraph=True)
        def f(a, b):
            xs = b.tolist()
            for x in xs:
                torch._check_is_size(x)
                torch._check(x <= self.N)
            return a.split(xs)

        f(self.input, self.splits)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
