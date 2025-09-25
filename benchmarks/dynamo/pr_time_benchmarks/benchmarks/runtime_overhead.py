import sys

from benchmark_base import BenchmarkBase

import torch
from torch.autograd.grad_mode import inference_mode


class Benchmark(BenchmarkBase):
    def __init__(self, requires_grad, inference_mode, backward, dynamic):
        assert not (inference_mode and backward), (
            "inference_mode and backward cannot be both True"
        )

        self._requires_grad = requires_grad
        self._inference_mode = inference_mode
        self._backward = backward

        super().__init__(
            category="runtime_overhead",
            backend="inductor",
            device="cuda",
            dynamic=dynamic,
        )

    def name(self):
        prefix = f"{self.category()}_{self.backend()}"
        if self._requires_grad:
            prefix += "_requires_grad"
        if self._inference_mode:
            prefix += "_inference_mode"
        if self._backward:
            prefix += "_backward"
        if self.is_dynamic():
            prefix += "_dynamic"
        return prefix

    def description(self):
        return "runtime of a compiled add1 op small input"

    def _prepare_once(self):
        torch._dynamo.reset()
        self.a = torch.ones(2, device=self.device(), requires_grad=self._requires_grad)

        @torch.compile(
            backend=self.backend(),
            fullgraph=True,
            dynamic=self.is_dynamic(),
        )
        def add1(a):
            return a + 1

        self._add1 = add1

        # warmup
        for _ in range(10):
            if self._backward:
                self.forward_val = self._add1(self.a).sum()
                self.forward_val.backward()
            else:
                self._work()

    def _prepare(self):
        if self._backward:
            self.forward_val = self._add1(self.a).sum()

    def _work(self):
        if self._inference_mode:
            with inference_mode():
                self._add1(self.a)
        elif self._backward:
            self.forward_val.backward()
        else:
            self._add1(self.a)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(
            requires_grad=False, inference_mode=False, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=False, inference_mode=True, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=True, dynamic=False
        ),
        Benchmark(
            requires_grad=False, inference_mode=False, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=False, inference_mode=True, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=True, dynamic=True
        ),
    ]

    for benchmark in all:
        benchmark.enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
