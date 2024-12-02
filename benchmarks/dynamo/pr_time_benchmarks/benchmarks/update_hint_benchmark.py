import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    N = 20

    def __init__(self):
        super().__init__(
            category="update_hint",
            backend="inductor",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}_regression"

    def description(self):
        return "information at https://github.com/pytorch/pytorch/pull/129893"

    def _prepare_once(self):
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(0)

        self.splits = torch.randint(10, (self.N,))
        sz = self.splits.sum().item()
        self.input = torch.randn(sz)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
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
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
