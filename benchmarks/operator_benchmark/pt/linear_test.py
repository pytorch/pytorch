from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench 
import torch
import torch.nn as nn


"""Microbenchmarks for Linear operator."""

configs = op_bench.config_list(
    attrs=[
        [1, 32, 10],
        [4, 256, 100],
        [16, 1024, 256],
    ],
    attr_names=["N", "IN", "OUT"],
    tags=["short"]
)


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT):
        self.input_one = torch.rand(N, IN)
        self.linear = nn.Linear(IN, OUT) 
        self.set_module_name("linear")

    def forward(self):
        return self.linear(self.input_one)


op_bench.generate_pt_test(configs, LinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
