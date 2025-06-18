import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_inductor_cache


class Benchmark(BenchmarkBase):
    def __init__(self, backend, dynamic=False, is_gpu=False):
        super().__init__(
            category="add_loop",
            backend=backend,
            device="cuda" if is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        prefix = f"{self.category()}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self.device() == "cuda":
            prefix += "_gpu"
        return prefix

    def description(self):
        return "a loop over 100 add node"

    def _prepare_once(self):
        self.a = torch.ones(1000, device=self.device())
        self.b = torch.torch.ones(1000, device=self.device())

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(
            backend=self.backend(),
            fullgraph=True,
            dynamic=self.is_dynamic(),
        )
        def f(a, b):
            result = a.clone()
            for i in range(1000):
                if i % 3 == 0:
                    result = result + b
                elif i % 3 == 1:
                    result = result + 8 * b
                else:
                    result = result.sin()
            return result

        with fresh_inductor_cache():
            f(self.a, self.b)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark("eager"),
        Benchmark("eager", dynamic=True),
        Benchmark("inductor"),
        Benchmark("inductor", is_gpu=True),
        Benchmark("inductor", is_gpu=True, dynamic=True),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
