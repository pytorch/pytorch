from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    tags=["short"],
)


class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims):
        self.X = (torch.rand(*dims) - 0.5) * 256
        self.weight = torch.rand(*self.X.size()[1:], dtype=torch.float)
        self.bias = torch.rand(*self.X.size()[1:], dtype=torch.float)
        self.eps = 1e-5

    def forward(self):
        return F.layer_norm(
            self.X, self.X.size()[1:], weight=self.weight, bias=self.bias, eps=self.eps)


op_bench.generate_pt_test(layernorm_configs_short, LayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
