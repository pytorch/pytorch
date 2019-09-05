from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Cat operator"""


# Configs for PT Cat operator
cat_short_configs = op_bench.cross_product_configs(
    M=[256, 512],
    N=[512],
    tags=['short']
)


class CatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N):
        self.input_one = torch.rand(M, N)
        self.set_module_name('cat')

    def forward(self):
        return torch.cat((self.input_one, self.input_one))


op_bench.generate_pt_test(cat_short_configs, CatBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
