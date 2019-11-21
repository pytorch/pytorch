from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Cat operator"""


# Configs for PT Cat operator
cat_configs_short = op_bench.config_list(
    attr_names=['M', 'N', 'K', 'dim'],
    attrs=[
        [256, 512, 1, 0],
        [512, 512, 2, 1],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short'],
)

cat_configs_long = op_bench.cross_product_configs(
    M=[128],
    N=[128, 1024],
    K=[1, 2],
    dim=[0, 1, 2],
    device=['cpu', 'cuda'],
    tags=['long']
)


class CatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, device):
        self.input_one = torch.rand(M, N, K, device=device)
        self.dim = dim
        self.set_module_name('cat')

    def forward(self):
        return torch.cat((self.input_one, self.input_one), dim=self.dim)


op_bench.generate_pt_test(cat_configs_short + cat_configs_long,
                          CatBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
