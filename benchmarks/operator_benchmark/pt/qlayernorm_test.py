from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch


"""Microbenchmarks for quantized layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QLayerNormBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, dims, dtype):
        X = (torch.rand(*dims) - 0.5) * 256
        scale = 1.0
        zero_point = 0
        self.qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=dtype)
        self.weight = torch.rand(*self.qX.size()[1:], dtype=torch.float)
        self.bias = torch.rand(*self.qX.size()[1:], dtype=torch.float)
        self.eps = 1e-5
        self.Y_scale = 0.1
        self.Y_zero_point = 0

    def forward(self):
        return torch.ops.quantized.layer_norm(
            self.qX, self.qX.size()[1:], weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.Y_scale,
            output_zero_point=self.Y_zero_point)


op_bench.generate_pt_test(layernorm_configs_short, QLayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
