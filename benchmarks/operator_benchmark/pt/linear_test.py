from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn as nn

from . import configs


"""Microbenchmarks for Linear operator."""


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, device):
        self.input_one = torch.rand(N, IN, device=device)
        self.linear = nn.Linear(IN, OUT).to(device=device)
        self.set_module_name("linear")

    def forward(self):
        return self.linear(self.input_one)


op_bench.generate_pt_test(configs.linear_configs_short + configs.linear_configs_long,
                          LinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
