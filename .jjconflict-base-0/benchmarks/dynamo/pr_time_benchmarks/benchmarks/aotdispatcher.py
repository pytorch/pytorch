import sys

from benchmark_base import BenchmarkBase

import torch
from torch.testing._internal.two_tensor import TwoTensor


class Benchmark(BenchmarkBase):
    def __init__(self, *, training, subclass):
        self._training = training
        self._subclass = subclass
        super().__init__(
            category="aotdispatcher",
            backend="aot_eager_decomp_partition",
            device="cpu",
            mode="training" if self._training else "inference",
        )

    def name(self):
        prefix = f"{self.category()}_{self.mode()}"
        if self._subclass:
            prefix += "_subclass"
        else:
            prefix += "_nosubclass"
        if self.device() == "cpu":
            prefix += "_cpu"
        return prefix

    def description(self):
        return "100 inputs, 100 outputs, each input is added once"

    def _prepare_once(self):
        _args = [
            torch.ones(100, requires_grad=self._training, device=self.device())
            for _ in range(100)
        ]
        if self._subclass:
            _args = [
                TwoTensor(x, x.clone().detach().requires_grad_(self._training))
                for x in _args
            ]
        self._args = _args

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(*args):
            outs = [torch.add(x, x) for x in args]
            return outs

        f(*self._args)


def main():
    result_path = sys.argv[1]
    all = [
        Benchmark(training=False, subclass=False),
        Benchmark(training=True, subclass=False),
        Benchmark(training=False, subclass=True),
        Benchmark(training=True, subclass=True),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
