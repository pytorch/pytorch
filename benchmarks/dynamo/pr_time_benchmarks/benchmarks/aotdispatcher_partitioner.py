import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    def __init__(self):
        self._model_type = "aotdispatcher_partitioner"
        self._backend = "aot_eager_decomp_partition"
        self._device = "cpu"

    def name(self):
        return f"{self.model_type()}_{self.device()}"

    def backend(self):
        return self._backend

    def model_type(self):
        return self._model_type

    def device(self):
        return self._device

    def is_fullgraph(self):
        return True

    def description(self):
        return "partitioner benchmark 1 input and 100 weights, mix of recompute and non-recompute ops"

    def _prepare_once(self):
        self.weights = [torch.randn(16, 16, requires_grad=True) for _ in range(100)]
        self.inp = torch.randn(16, 16)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(inp, *weights):
            x = inp
            for w in weights:
                x = torch.matmul(w, x).sin().sin()
            return x

        f(self.inp, *self.weights)


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
