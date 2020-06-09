from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch

"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

add_short_configs = op_bench.config_list(
    attr_names=['M', 'N', 'K'], 
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
        'dtype': [torch.float, torch.float64],
    },
    tags=['short'], 
)


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype): 
        self.input_one = torch.rand(M, N, K, device=device, dtype=dtype, requires_grad=True)
        self.input_two = torch.rand(M, N, K, device=device, dtype=dtype)
        self.set_module_name('add')

    def forward(self):
        return torch.add(self.input_one, self.input_two)


op_bench.generate_pt_test(add_short_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
