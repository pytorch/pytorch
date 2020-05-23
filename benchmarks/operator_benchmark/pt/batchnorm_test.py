from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for batchnorm operator."""

batchnorm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 256, 3136],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=["short"]
)

batchnorm_configs_long = op_bench.cross_product_configs(
    M=[1, 128],
    N=[8192, 2048],
    K=[1],
    device=['cpu', 'cuda'],
    tags=["long"]
)


class BatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.input_one = torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        self.mean = torch.rand(N, device=device)
        self.var = torch.rand(N, device=device)
        self.weight = torch.rand(N, device=device)
        self.bias = torch.rand(N, device=device)
        self.set_module_name("batchnorm")

    def forward(self):
        return F.batch_norm(self.input_one, self.mean, self.var, self.weight, self.bias)


op_bench.generate_pt_test(batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark)
op_bench.generate_pt_gradient_test(batchnorm_configs_short + batchnorm_configs_long, BatchNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
