import random

import torch

from ..benchmark_base import Benchmark


class P0(Benchmark):
    N = 20

    def name(self):
        return "update_hint_regression"

    def description(self):
        return "information at https://github.com/pytorch/pytorch/pull/129893"

    def reset(self):
        pass

    def prepare(self):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        random.seed(42)
        self.splits = torch.randint(10, (self.N,))
        sz = self.splits.sum().item()
        self.input = torch.randn(sz)

    def work(self):
        @torch.compile(fullgraph=True)
        def f(a, b):
            xs = b.tolist()
            for x in xs:
                torch._check_is_size(x)
                torch._check(x <= self.N)
            return a.split(xs)

        f(self.input, self.splits)


def main():
    print(P0().enable_instruction_count().collect_all())


if __name__ == "__main__":
    main()
