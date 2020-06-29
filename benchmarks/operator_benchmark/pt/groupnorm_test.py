from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for groupnorm operator."""

groupnorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    num_groups=(2, 4),
    tags=["short"],
)


class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, num_groups):
        self.X = (torch.rand(*dims) - 0.5) * 256
        self.num_groups = num_groups
        num_channels = dims[1]
        self.weight = torch.rand(num_channels, dtype=torch.float)
        self.bias = torch.rand(num_channels, dtype=torch.float)
        self.eps = 1e-5

    def forward(self):
        return F.group_norm(
            self.X, self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)


op_bench.generate_pt_test(groupnorm_configs_short, GroupNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
