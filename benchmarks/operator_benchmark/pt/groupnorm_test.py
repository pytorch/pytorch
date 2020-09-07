from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools as it

import operator_benchmark as op_bench
import torch
import torch.nn.functional as F

from . import configs


"""Microbenchmarks for groupnorm operator."""

_group_configs = [[{"num_groups": n}] for n in [2, 4]]
groupnorm_configs = [
    group_cfg + norm_cfg for group_cfg, norm_cfg in
    it.product(_group_configs, configs.norm_fuzzed_configs)

    # Group Norm requires that num_groups evenly divides num_channels
    if not (norm_cfg[0]["X_SIZE"][1] % group_cfg[0]["num_groups"])
]

class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, num_groups, device):
        self.X = (torch.rand(X_SIZE, device=device) - 0.5) * 256
        self.X.requires_grad_(requires_grad=self.auto_set())
        self.num_groups = num_groups
        num_channels = X_SIZE[1]
        self.weight = torch.rand(num_channels, dtype=torch.float, device=device)
        self.bias = torch.rand(num_channels, dtype=torch.float, device=device)
        self.eps = 1e-5

    def forward(self):
        return F.group_norm(
            self.X, self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)

op_bench.generate_pt_test(groupnorm_configs, GroupNormBenchmark)
op_bench.generate_pt_gradient_test(groupnorm_configs, GroupNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
