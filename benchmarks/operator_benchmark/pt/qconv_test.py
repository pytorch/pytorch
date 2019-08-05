from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn.quantized as nnq


"""
Microbenchmarks for qConv2d operators.
"""

# Configs for qconv2d
qconv_2d_configs = op_bench.config_list(
    # Resnext101 - 32x8d shapes
    attrs=[
        [1, 64, 128, 56, 56, 1, 1, 1, 0],
        [1, 256, 256, 56, 56, 32, 3, 1, 1],
        [1, 256, 256, 56, 56, 1, 1, 1, 0],
        [1, 512, 512, 56, 56, 32, 3, 2, 1],
    ],
    attr_names=[
        "N", "IC", "OC", "H", "W", "G", "kernel", "stride", "pad"
    ],
    tags=["short"],
)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        X = X.permute([0, 2, 3, 1]).contiguous()
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        W = W.permute([0, 2, 3, 1]).contiguous()
        qW = torch.quantize_linear(W, scale=scale, zero_point=0, dtype=torch.qint8)

        self.input = qX
        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.weight = qW
        self.qconv2d.scale = scale
        self.qconv2d.zero_point = zero_point
        self.set_module_name("QConv2d")

    def forward(self):
        return self.qconv2d(self.input)


op_bench.generate_pt_test(qconv_2d_configs, QConv2dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
