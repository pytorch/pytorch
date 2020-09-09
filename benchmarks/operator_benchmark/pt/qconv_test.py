from __future__ import absolute_import, division, print_function, unicode_literals

import operator_benchmark as op_bench
import torch
import torch.nn.quantized as nnq

from . import configs

"""
Microbenchmarks for qConv operators.
"""

class QConv1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, device):
        G = 1
        pad = 0
        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(X_SIZE, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(C_OUT, X_SIZE[1] // G, KERNEL_SIZE, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(W, scale=self.scale, zero_point=0, dtype=torch.qint8)

        self.input = qX

        self.qconv1d = nnq.Conv1d(X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE, padding=pad, groups=G)
        self.qconv1d.set_weight_bias(self.qW, None)
        self.qconv1d.scale = torch.tensor([self.scale], dtype=torch.double)
        self.qconv1d.zero_point = torch.tensor([self.zero_point], dtype=torch.int)
        self.set_module_name("QConv1d")

    def forward(self):
        return self.qconv1d(self.input)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    # def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, GROUPS, device):
        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(X_SIZE, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, X_SIZE[1] // G, KERNEL_SIZE, KERNEL_SIZE, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(W, scale=self.scale, zero_point=0, dtype=torch.qint8)

        self.input = qX

        self.qconv2d = nnq.Conv2d(X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE, padding=pad, groups=G)
        self.qconv2d.set_weight_bias(self.qW, None)
        self.qconv2d.scale = torch.tensor([self.scale], dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor([self.zero_point], dtype=torch.int)
        self.set_module_name("QConv2d")

    def forward(self):
        return self.qconv2d(self.input)


op_bench.generate_pt_test(configs.remove_cuda(configs.conv1d_fuzzed_configs_short + configs.conv1d_fuzzed_configs_long), QConv1dBenchmark)
op_bench.generate_pt_test(configs.remove_cuda(configs.conv2d_fuzzed_configs_short + configs.conv2d_fuzzed_configs_long), QConv2dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
