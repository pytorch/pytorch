import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    N = 100

    def name(self):
        return "sum_floordiv_regression"

    def description(self):
        return "information at https://github.com/pytorch/pytorch/issues/134133"

    def _prepare_once(self):
        class M(torch.nn.Module):
            def forward(self, x):
                total = sum(t.item() for t in x)
                return total // 2

        self.m = M()
        self.input = [torch.tensor(i + 2) for i in range(self.N)]

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        torch.export.export(self.m, (self.input,))


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
