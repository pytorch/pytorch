import sys

from benchmark_base import BenchmarkBase

import torch


_OUTPUTS = None


class NoSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
        return _OUTPUTS

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("forward-only benchmark")


class Benchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(category="autograd_function_apply", device="cuda")

    def name(self):
        return f"{self.category()}_13_in_2_out_no_save"

    def description(self):
        return "autograd.Function.apply with 13 inputs, 2 outputs, and no saved tensors"

    def _prepare_once(self):
        global _OUTPUTS
        _OUTPUTS = (
            torch.empty((0,), device=self.device()),
            torch.empty((0,), device=self.device()),
        )
        self._args = tuple(
            torch.empty((0,), device=self.device(), requires_grad=True)
            for _ in range(13)
        )
        for _ in range(10000):
            NoSave.apply(*self._args)
        torch.cuda.synchronize()

    def _prepare(self):
        pass

    def _work(self):
        NoSave.apply(*self._args)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
