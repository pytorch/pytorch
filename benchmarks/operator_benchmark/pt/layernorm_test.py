from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F

from . import configs


"""Microbenchmarks for layernorm operator."""
class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, device):
        self.X = (torch.rand(X_SIZE, device=device) - 0.5) * 256
        self.X.requires_grad_(requires_grad=self.auto_set())
        self.weight = torch.rand(X_SIZE[1:], dtype=torch.float, device=device)
        self.bias = torch.rand(X_SIZE[1:], dtype=torch.float, device=device)
        self.eps = 1e-5

    def forward(self):
        return F.layer_norm(
            self.X, self.X.size()[1:], weight=self.weight, bias=self.bias, eps=self.eps)


op_bench.generate_pt_test(configs.norm_fuzzed_configs, LayerNormBenchmark)
op_bench.generate_pt_gradient_test(configs.norm_fuzzed_configs, LayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
