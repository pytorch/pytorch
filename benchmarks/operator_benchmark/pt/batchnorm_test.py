from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F

from . import configs


"""Microbenchmarks for batchnorm operator."""
class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, device):
        n = X_SIZE[1]
        self.input_one = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.mean = torch.rand(n, device=device)
        self.var = torch.rand(n, device=device)
        self.weight = torch.rand(n, device=device)
        self.bias = torch.rand(n, device=device)
        self.set_module_name("batchnorm")

    def forward(self):
        return F.batch_norm(self.input_one, self.mean, self.var, self.weight, self.bias)

op_bench.generate_pt_test(configs.norm_fuzzed_configs_short + configs.norm_fuzzed_configs_long, BatchNormBenchmark)
op_bench.generate_pt_gradient_test(configs.norm_fuzzed_configs_short + configs.norm_fuzzed_configs_long, BatchNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
