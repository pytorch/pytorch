from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""
Microbenchmarks for Conv1d and ConvTranspose1d operators.
"""


# Configs for conv-1d ops
conv_1d_configs = op_bench.config_list(
    attrs=[
        [16, 33, 3, 1, 1, 64],
        [16, 33, 3, 2, 16, 128],
    ],
    attr_names=[
        "in_c", "out_c", "kernel", "stride", "N", "L"
    ],
    tags=["short"]
)


class Conv1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, L):
        self.input = torch.rand(N, in_c, L) 
        self.conv1d = nn.Conv1d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("Conv1d")

    def forward(self):
        return self.conv1d(self.input)


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, L):
        self.input = torch.rand(N, in_c, L) 
        self.convtranspose1d = nn.ConvTranspose1d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("ConvTranspose1d")

    def forward(self):
        return self.convtranspose1d(self.input)


op_bench.generate_pt_test(conv_1d_configs, Conv1dBenchmark)
op_bench.generate_pt_test(conv_1d_configs, ConvTranspose1dBenchmark)


"""
Microbenchmarks for Conv2d and ConvTranspose2d operators.
"""


# Configs for Conv2d and ConvTranspose1d
conv_2d_configs = op_bench.config_list(
    attrs=[
        [16, 33, 3, 1, 1, 32, 32],
        [16, 33, 3, 2, 16, 64, 64],
    ],
    attr_names=[
        "in_c", "out_c", "kernel", "stride", "N", "H", "W"
    ],
    tags=["short"]
)


class Conv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, H, W):
        self.input = torch.rand(N, in_c, H, W) 
        self.conv2d = nn.Conv2d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("Conv2d")

    def forward(self):
        return self.conv2d(self.input)


class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, H, W):
        self.input = torch.rand(N, in_c, H, W) 
        self.convtranspose2d = nn.ConvTranspose2d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("ConvTranspose2d")

    def forward(self):
        return self.convtranspose2d(self.input)


op_bench.generate_pt_test(conv_2d_configs, Conv2dBenchmark)
op_bench.generate_pt_test(conv_2d_configs, ConvTranspose2dBenchmark)


"""
Microbenchmarks for Conv3d and ConvTranspose3d operators.
"""

# Configs for Conv3d and ConvTranspose3d
conv_3d_configs = op_bench.config_list(
    attrs=[
        [16, 33, 3, 1, 8, 4, 32, 32],
        [16, 33, 3, 2, 16, 8, 64, 64],
    ],
    attr_names=[
        "in_c", "out_c", "kernel", "stride", "N", "D", "H", "W"
    ],
    tags=["short"]
)


class Conv3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, D, H, W):
        self.input = torch.rand(N, in_c, D, H, W) 
        self.conv3d = nn.Conv3d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("Conv3d")

    def forward(self):
        return self.conv3d(self.input)


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, D, H, W):
        self.input = torch.rand(N, in_c, D, H, W) 
        self.convtranspose3d = nn.ConvTranspose3d(in_c, out_c, kernel, stride=stride)
        self.set_module_name("ConvTranspose3d")

    def forward(self):
        return self.convtranspose3d(self.input)


op_bench.generate_pt_test(conv_3d_configs, Conv3dBenchmark)
op_bench.generate_pt_test(conv_3d_configs, ConvTranspose3dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
