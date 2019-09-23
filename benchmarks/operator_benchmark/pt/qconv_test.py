from __future__ import absolute_import, division, print_function, unicode_literals

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
    attr_names=["N", "IC", "OC", "H", "W", "G", "kernel", "stride", "pad"],
    tags=["short"],
)

# Configs for convolution shapes from Resnext-101 32x4d
resnext_32_4d_shape_configs = op_bench.config_list(
    attrs=[
        [1, 1024, 1024, 14, 14, 1, 1, 1, 0],  # op#312
        [1, 1024, 1024, 14, 14, 32, 3, 2, 1],  # op#315
        [1, 1024, 2048, 14, 14, 1, 1, 2, 0],  # op#320
        [1, 1024, 512, 14, 14, 1, 1, 1, 0],  # op#92
        [1, 1024, 1024, 7, 7, 32, 3, 1, 1],  # op#327
        [1, 1024, 2048, 7, 7, 1, 1, 1, 0],  # op#318
        [1, 128, 128, 56, 56, 32, 3, 1, 1],  # op#9
        [1, 128, 256, 56, 56, 1, 1, 1, 0],  # op#12
        [1, 2048, 1024, 7, 7, 1, 1, 1, 0],  # op#324
        [1, 256, 256, 28, 28, 32, 3, 1, 1],  # op#53
        [1, 256, 512, 28, 28, 1, 1, 1, 0],  # op#44
        [1, 256, 128, 56, 56, 1, 1, 1, 0],  # op#28
        [1, 256, 128, 56, 56, 1, 1, 1, 1],  # op#18
        [1, 256, 256, 56, 56, 1, 1, 1, 0],  # op#38
        [1, 256, 256, 56, 56, 32, 3, 2, 1],  # op#41
        [1, 256, 512, 56, 56, 1, 1, 2, 0],  # op#46
        [1, 3, 64, 224, 224, 1, 7, 2, 3],  # op#2
        [1, 512, 1024, 14, 14, 1, 1, 1, 0],  # op#86
        [1, 512, 512, 14, 14, 32, 3, 1, 1],  # op#95
        [1, 512, 1024, 28, 28, 1, 1, 2, 0],  # op#88
        [1, 512, 256, 28, 28, 1, 1, 1, 0],  # op#50
        [1, 512, 512, 28, 28, 1, 1, 1, 0],  # op#80
        [1, 512, 512, 28, 28, 32, 3, 2, 1],  # op#83
        [1, 64, 128, 56, 56, 1, 1, 1, 0],  # op#6
        [1, 64, 256, 56, 56, 1, 1, 1, 0],  # op#14
    ],
    attr_names=["N", "IC", "OC", "H", "W", "G", "kernel", "stride", "pad"],
    tags=["resnext101_32x4d"],
)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        self.input = qX
        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.weight = qW
        self.qconv2d.scale = torch.tensor([scale], dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor([zero_point], dtype=torch.int)
        self.set_module_name("QConv2d")

    def forward(self):
        return self.qconv2d(self.input)


class QConv2dChainedBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
        scale = 1.0 / 255
        zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        self.input = qX
        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.weight = qW
        self.qconv2d.scale = torch.tensor([scale], dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor([zero_point], dtype=torch.int)

        W2 = torch.randn(OC, OC // G, kernel, kernel, dtype=torch.float32)
        qW2 = torch.quantize_per_tensor(W2, scale=scale, zero_point=0, dtype=torch.qint8)
        self.qconv2d2 = nnq.Conv2d(OC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d2.weight = qW2
        self.qconv2d2.scale = torch.tensor([scale], dtype=torch.double)
        self.qconv2d2.zero_point = torch.tensor([zero_point], dtype=torch.int)
        self.set_module_name("QConv2dChained")

    def forward(self):
        # test that layout propagation works fine
        x = self.qconv2d(self.input)
        x = x.relu()
        return self.qconv2d2(x)


op_bench.generate_pt_test(qconv_2d_configs, QConv2dBenchmark)
op_bench.generate_pt_test(resnext_32_4d_shape_configs, QConv2dBenchmark)
op_bench.generate_pt_test(qconv_2d_configs, QConv2dChainedBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
