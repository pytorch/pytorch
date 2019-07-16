from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.quantized as nnq


"""
Microbenchmarks for qlinear operators.
"""

# Configs for qlinear
qlinear_configs = op_bench.config_list(
    # Resnext101 - 32x8d shapes
    attrs=[
        [1, 32, 10],
        [4, 256, 100],
        [16, 1024, 256],
    ],
    attr_names=["N", "IN", "OUT"],
    tags=["short"],
)


class QLinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, IN, dtype=torch.float32)
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        W = torch.randn(OUT, IN, dtype=torch.float32)
        qW = torch.quantize_linear(W, scale=scale, zero_point=0, dtype=torch.qint8)

        self.input = qX
        self.qlinear = nnq.Linear(IN, OUT)
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point
        self.set_module_name("QLinear")

    def forward(self):
        return self.qlinear(self.input)


op_bench.generate_pt_test(qlinear_configs, QLinearBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
