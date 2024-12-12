import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(
            category="float_args",
            backend="inductor",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}"

    def description(self):
        return "Benchmark to measure recompilations with random float arguments."

    def _prepare_once(self):
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(0)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend="inductor")
        def f(x, y):
            return x + y

        for i in range(8):
            f(torch.randn(3), i * 2.5)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
