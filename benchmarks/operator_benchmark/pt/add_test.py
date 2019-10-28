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
    N=range(2, 128, 64),
    K=[8 ** x for x in range(0, 3)], 
    device=['cpu'],
    tags=["long"]
)


add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"], 
    attrs=[
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        'device': ['cpu'],
    },
    tags=["short"], 
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device): 
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        self.input_two = torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        self.set_module_name("add")

    def forward(self):
        return torch.add(self.input_one, self.input_two)

# The generated test names based on add_short_configs will be in the following pattern: 
# add_M8_N16_K32_devicecpu
# add_M8_N16_K32_devicecpu_bwdall
# add_M8_N16_K32_devicecpu_bwd1
# add_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests. 

op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
