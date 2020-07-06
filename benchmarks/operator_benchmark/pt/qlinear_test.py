from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench

import torch
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd


"""
Microbenchmarks for Quantized Linear operators.
"""

# Configs for qlinear
qlinear_configs = op_bench.config_list(
    attrs=[
        # matches floating point liner
        [1, 1, 1],
        [4, 256, 128],
        [16, 512, 256],
        # other
        [1024, 1024, 1024],
        [64, 320, 800],
        [64, 512, 768],
        [16, 512, 256],
        [128, 128, 128],
        [256, 256, 512],
        [6400, 141, 15],
        [6400, 141, 8],
        [16, 2504, 211],
        [16, 1434, 369],
        [1, 3496, 1024],
        [16, 512, 256],
        [1, 3456, 1600],
    ],
    attr_names=["N", "IN", "OUT"],  # M, K, N
    tags=["short"],
)


class _QLinearBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, linear_under_test):
        scale = torch.tensor(1.0 / 255)
        zero_point = torch.tensor(0)
        self.X = torch.randn(N, IN, dtype=torch.float32)
        self.qX = torch.quantize_per_tensor(self.X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        W = torch.randn(OUT, IN, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        # Assume that the `self.qlinear` is set in the child
        self.qlinear = linear_under_test
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point

    def forward(self):
        # Assume that the `self.input` is set in the child
        return self.qlinear(self.input)

class QLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT):
        super(QLinearBenchmark, self).init(N, IN, OUT, nnq.Linear(IN, OUT))
        self.input = self.qX
        self.set_module_name("QLinear")


class QDynamicLinearBenchmark(_QLinearBenchmarkBase):
    def init(self, N, IN, OUT):
        super(QDynamicLinearBenchmark, self).init(N, IN, OUT, nnqd.Linear(IN, OUT))
        self.input = self.X
        self.set_module_name("QDynamicLinear")


op_bench.generate_pt_test(qlinear_configs, QLinearBenchmark)
op_bench.generate_pt_test(qlinear_configs, QDynamicLinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
