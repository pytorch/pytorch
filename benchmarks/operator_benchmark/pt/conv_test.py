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
conv_1d_configs_short = op_bench.config_list(
    attr_names=[
        'in_c', 'out_c', 'kernel', 'stride', 'N', 'L'
    ],
    attrs=[
        [128, 256, 3, 1, 1, 64],
        [256, 256, 3, 2, 4, 64],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)

conv_1d_configs_long = op_bench.cross_product_configs(
    in_c=[128, 512],
    out_c=[128, 512],
    kernel=[3],
    stride=[1, 2],
    N=[8],
    L=[128],
    device=['cpu', 'cuda'],
    tags=["long"]
)


class Conv1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, L, device):
        self.input = torch.rand(N, in_c, L, device=device)
        self.conv1d = nn.Conv1d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('Conv1d')

    def forward(self):
        return self.conv1d(self.input)


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, L, device):
        self.input = torch.rand(N, in_c, L, device=device)
        self.convtranspose1d = nn.ConvTranspose1d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('ConvTranspose1d')

    def forward(self):
        return self.convtranspose1d(self.input)


op_bench.generate_pt_test(conv_1d_configs_short + conv_1d_configs_long,
                          Conv1dBenchmark)
op_bench.generate_pt_test(conv_1d_configs_short + conv_1d_configs_long,
                          ConvTranspose1dBenchmark)


"""
Microbenchmarks for Conv2d and ConvTranspose2d operators.
"""


# Configs for Conv2d and ConvTranspose1d
conv_2d_configs_short = op_bench.config_list(
    attr_names=[
        'in_c', 'out_c', 'kernel', 'stride', 'N', 'H', 'W'
    ],
    attrs=[
        [256, 256, 3, 1, 1, 16, 16],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)

conv_2d_configs_long = op_bench.cross_product_configs(
    in_c=[128, 256],
    out_c=[128, 256],
    kernel=[3],
    stride=[1, 2],
    N=[4],
    H=[32],
    W=[32],
    device=['cpu', 'cuda'],
    tags=["long"]
)


class Conv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, H, W, device):
        self.input = torch.rand(N, in_c, H, W, device=device)
        self.conv2d = nn.Conv2d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('Conv2d')

    def forward(self):
        return self.conv2d(self.input)


class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, H, W, device):
        self.input = torch.rand(N, in_c, H, W, device=device)
        self.convtranspose2d = nn.ConvTranspose2d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('ConvTranspose2d')

    def forward(self):
        return self.convtranspose2d(self.input)


op_bench.generate_pt_test(conv_2d_configs_short + conv_2d_configs_long,
                          Conv2dBenchmark)
op_bench.generate_pt_test(conv_2d_configs_short + conv_2d_configs_long,
                          ConvTranspose2dBenchmark)


"""
Microbenchmarks for Conv3d and ConvTranspose3d operators.
"""

# Configs for Conv3d and ConvTranspose3d
conv_3d_configs_short = op_bench.config_list(
    attr_names=[
        'in_c', 'out_c', 'kernel', 'stride', 'N', 'D', 'H', 'W'
    ],
    attrs=[
        [64, 64, 3, 1, 8, 4, 16, 16],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short']
)


class Conv3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, D, H, W, device):
        self.input = torch.rand(N, in_c, D, H, W, device=device)
        self.conv3d = nn.Conv3d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('Conv3d')

    def forward(self):
        return self.conv3d(self.input)


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_c, out_c, kernel, stride, N, D, H, W, device):
        self.input = torch.rand(N, in_c, D, H, W, device=device)
        self.convtranspose3d = nn.ConvTranspose3d(in_c, out_c, kernel, stride=stride).to(device=device)
        self.set_module_name('ConvTranspose3d')

    def forward(self):
        return self.convtranspose3d(self.input)


op_bench.generate_pt_test(conv_3d_configs_short, Conv3dBenchmark)
op_bench.generate_pt_test(conv_3d_configs_short,
                          ConvTranspose3dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
