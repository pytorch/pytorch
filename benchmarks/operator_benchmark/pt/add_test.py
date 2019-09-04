from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator 
add_long_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)], 
    tags=["long"]
)


add_short_configs = op_bench.config_list(
    attrs=[
        [8, 16, 32],
        [16, 32, 64],
    ],
    attr_names=["M", "N", "K"], 
    tags=["short"], 
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K): 
        self.input_one = torch.rand(M, N, K)
        self.input_two = torch.rand(M, N, K)
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
