from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch


"""Microbenchmarks for quantized instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QInstanceNormBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, dims, dtype):
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0
        self.qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=dtype)
        self.weight = torch.rand(num_channels, dtype=torch.float)
        self.bias = torch.rand(num_channels, dtype=torch.float)
        self.eps = 1e-5
        self.Y_scale = 0.1
        self.Y_zero_point = 0

    def forward(self):
        return torch.ops.quantized.instance_norm(
            self.qX, weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.Y_scale,
            output_zero_point=self.Y_zero_point)


op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
