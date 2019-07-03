from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for Split operator"""


# Configs for PT Split operator
split_short_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    parts=[2, 3],
    tags=["short"]
)


class SplitBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, parts): 
        self.input_one = torch.rand(M, N) 
        self.split_size = int(M * N / parts)
        self.set_module_name("split")

    def forward(self):
        return torch.split(self.input_one, self.split_size)


op_bench.generate_pt_test(split_short_configs, SplitBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
