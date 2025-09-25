import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_inductor_cache
from torch.autograd.grad_mode import inference_mode


class Benchmark(BenchmarkBase):
    def __init__(self, requires_grad, inference_mode):
        self._requires_grad = requires_grad
        self._inference_mode = inference_mode

        super().__init__(
            category="runtime_overhead",
            backend="inductor",
            device="cuda",
            dynamic=False,
        )

    def name(self):
        prefix = f"{self.category()}_{self.backend()}"
        if self._requires_grad:
            prefix += "_requires_grad"
        if self._inference_mode:
            prefix += "_inference_mode"
        return prefix

    def description(self):
        return "runtime of a compiled add1 op small input"

    def _prepare_once(self):
        torch._dynamo.reset()
        self.a = torch.ones(1, device=self.device(), requires_grad=self._requires_grad)
        
        @torch.compile(
            backend=self.backend(),
            fullgraph=True,
            dynamic=self.is_dynamic(),
        )
        def add1(a):
            return a + 1
        
        self._add1 = add1

        # warmup
        self._work()

    def _prepare(self):
        pass

    def _work(self):
        if self._inference_mode:
            with inference_mode():
                self._add1(self.a)
        else:
            self._add1(self.a)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(False, False),
        Benchmark(False, True),
        Benchmark(True, False),
    ]

    for benchmark in all:
        benchmark.enable_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()