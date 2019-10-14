from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for as_strided operator"""


# Configs for PT as_strided operator
split_short_configs = op_bench.cross_product_configs(
    M=[256, 512],
    N=[256, 512],
    size=[(32, 32), (64, 64)],
    stride=[(1, 1), (2, 2)],
    storage_offset=[0, 1],
    tags=['short']
)


class As_stridedBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, size, stride, storage_offset):
        self.input_one = torch.rand(M, N)
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset
        self.set_module_name('as_strided')

    def forward(self):
        return torch.as_strided(
            self.input_one, self.size, self.stride, self.storage_offset)


op_bench.generate_pt_test(split_short_configs, As_stridedBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
