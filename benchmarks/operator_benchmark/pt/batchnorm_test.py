from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for batchnorm operator."""

configs = op_bench.config_list(
    attrs=[
        [1, 256, 3136],
        [1, 2 ** 16, 1],
        [128, 2048, 1],
    ],
    attr_names=["M", "N", "K"],
    tags=["short"]
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


op_bench.generate_pt_test(configs, BatchNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
