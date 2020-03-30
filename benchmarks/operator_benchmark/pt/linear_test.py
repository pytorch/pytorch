from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""Microbenchmarks for Linear operator."""

linear_configs_short = op_bench.config_list(
    attr_names=["N", "IN", "OUT"],
    attrs=[
        [1, 1, 1],
        [4, 256, 128],
        [16, 512, 256],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"]
)


linear_configs_long = op_bench.cross_product_configs(
    N=[32, 64],
    IN=[128, 512],
    OUT=[64, 128],
    device=['cpu', 'cuda'],
    tags=["long"]
)


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, device):
        self.input_one = torch.rand(N, IN, device=device)
        self.linear = nn.Linear(IN, OUT).to(device=device)
        self.set_module_name("linear")

    def forward(self):
        return self.linear(self.input_one)


op_bench.generate_pt_test(linear_configs_short + linear_configs_long,
                          LinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
