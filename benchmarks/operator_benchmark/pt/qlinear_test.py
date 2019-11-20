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
        [1024, 1024, 1024],
        [64, 800, 320],
        [64, 768, 512],
        [16, 256, 512],
        [128, 128, 128],
        [256, 512, 256],
        [6400, 15, 141],
        [6400, 8, 141],
        [16, 211, 2504],
        [16, 369, 1434],
        [1, 1024, 3496],
        [16, 256, 512],
        [1, 1600, 3456],
    ],
    attr_names=["N", "OUT", "IN"],  # M, N, K
    tags=["short"],
)


class _QLinearBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, IN, dtype=torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        W = torch.randn(OUT, IN, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        self.input = qX

        # Assume that the `self.qlinear` is set in the child
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point

    def forward(self):
        return self.qlinear(self.input)

class QLinearBenchmark(QLinearBenchmarkBase):
    def init(self, N, IN, OUT):
        self.qlinear = nnq.Linear(IN, OUT)
        self.set_module_name("QLinear")
        super(QLinearBenchmark, self).init(N, IN, OUT)


class QDynamicLinearBenchmark(QLinearBenchmarkBase):
    def init(self, N, IN, OUT):
        self.qlinear = nnqd.Linear(IN, OUT)
        self.set_module_name("QDynamicLinear")
        super(QDynamicLinearBenchmark, self).init(N, IN, OUT)


op_bench.generate_pt_test(qlinear_configs, QLinearBenchmark)
op_bench.generate_pt_test(qlinear_configs, QDynamicLinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
