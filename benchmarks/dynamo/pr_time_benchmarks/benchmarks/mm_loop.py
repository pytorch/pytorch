import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_inductor_cache


class Benchmark(BenchmarkBase):
    def __init__(self, is_dynamic: bool) -> None:
        super().__init__(
            category="mm_loop",
            backend="inductor",
            device="cuda",
            dynamic=is_dynamic,
        )

    def name(self) -> str:
        prefix = f"{self.category()}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self.device() == "cuda":
            prefix += "_gpu"
        return prefix

    def description(self) -> str:
        return "a mm 100 times in a loop with max auto tune on"

    def _prepare_once(self) -> None:
        self.a = torch.ones(10, 10, device=self.device())
        self.b = torch.torch.ones(10, 10, device=self.device())

    def _prepare(self) -> None:
        torch._dynamo.reset()

    def _work(self) -> None:
        @torch.compile(
            backend="inductor",
            fullgraph=True,
            dynamic=self._dynamic,
        )
        def f(a, b):
            z = torch.mm(a, b)
            for i in range(200):
                z = torch.mm(z, b)
            return z

        with fresh_inductor_cache(), torch._inductor.config.patch(max_autotune=True):
            f(self.a, self.b)


def main():
    result_path = sys.argv[1]
    all_benchamrks = [Benchmark(False), Benchmark(True)]
    for b in all_benchamrks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
