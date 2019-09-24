from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for batchnorm operator."""

configs_short = op_bench.config_list(
    attrs=[
        [1, 256, 3136],
    ],
    attr_names=["M", "N", "K"],
    tags=["short"]
)

configs_long = op_bench.cross_product_configs(
    M=[1, 128],
    N=[2 ** 16, 2048],
    K=[1],
    tags=["long"]
)


class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K):
        self.input_one = torch.rand(M, N, K)
        self.mean = torch.rand(N)
        self.var = torch.rand(N)
        self.weight = torch.rand(N)
        self.bias = torch.rand(N)
        self.set_module_name("batchnorm")

    def forward(self):
        return F.batch_norm(self.input_one, self.mean, self.var, self.weight, self.bias)


op_bench.generate_pt_test(configs_short + configs_long, BatchNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
