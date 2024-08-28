import gc
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    N = 20

    def name(self):
        return "add_loop"

    def description(self):
        return "a loop over 100 add node"

    def _prepare_once(self, dynamic=True):
        self.a = torch.ones(1000)
        self.b = torch.torch.ones(1000)

    def _prepare(self):
        torch._dynamo.reset()
        gc.collect()
        gc.disable()

    def _work(self):
        @torch.compile(fullgraph=True)
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

        f(self.a, self.b)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
