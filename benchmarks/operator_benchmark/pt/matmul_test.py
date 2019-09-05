from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for MatMul operator"""


# Configs for PT Matmul operator
mm_long_configs = op_bench.cross_product_configs(
    M=[64, 128, 256],
    N=range(2, 10, 3),
    K=[128, 512, 1024], 
    trans_a=[True, False],
    trans_b=[True, False],
    tags=["long"]
)


mm_short_configs = op_bench.config_list(
    attrs=[
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
    attr_names=["M", "N", "K", "trans_a", "trans_b"], 
    tags=["short"], 
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b): 
        self.input_one = torch.rand(M, N) if trans_a \
            else torch.rand(N, M).t()
        self.input_two = torch.rand(N, K) if trans_b else torch.rand(K, N).t()
        self.set_module_name("matmul")

    def forward(self):
        return torch.matmul(self.input_one, self.input_two)


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
