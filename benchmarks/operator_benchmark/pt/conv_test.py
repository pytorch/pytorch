from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench
import torch
import torch.nn as nn

from . import configs

"""Microbenchmarks for Conv1d and ConvTranspose1d operators."""
class Conv1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.conv1d = nn.Conv1d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE).to(device=device)
        self.set_module_name('Conv1d')

    def forward(self):
        return self.conv1d(self.input)


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.convtranspose1d = nn.ConvTranspose1d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE).to(device=device)
        self.set_module_name('ConvTranspose1d')

    def forward(self):
        return self.convtranspose1d(self.input)

conv1d_configs = configs.conv1d_fuzzed_configs_short + configs.conv1d_fuzzed_configs_long
op_bench.generate_pt_test(conv1d_configs, Conv1dBenchmark)
op_bench.generate_pt_test(conv1d_configs, ConvTranspose1dBenchmark)
op_bench.generate_pt_gradient_test(conv1d_configs, Conv1dBenchmark)
op_bench.generate_pt_gradient_test(conv1d_configs, ConvTranspose1dBenchmark)


"""Microbenchmarks for Conv2d and ConvTranspose2d operators."""
class Conv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, GROUPS, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.conv2d = nn.Conv2d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE, groups=GROUPS).to(device=device)
        self.set_module_name('Conv2d')

    def forward(self):
        return self.conv2d(self.input)

class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, GROUPS, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.convtranspose2d = nn.ConvTranspose2d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE, groups=GROUPS).to(device=device)
        self.set_module_name('ConvTranspose2d')

    def forward(self):
        return self.convtranspose2d(self.input)

conv2d_configs = configs.conv2d_fuzzed_configs_short + configs.conv2d_fuzzed_configs_long
op_bench.generate_pt_test(conv2d_configs, Conv2dBenchmark)
op_bench.generate_pt_test(conv2d_configs, ConvTranspose2dBenchmark)
op_bench.generate_pt_gradient_test(conv2d_configs, Conv2dBenchmark)
op_bench.generate_pt_gradient_test(conv2d_configs, ConvTranspose2dBenchmark)


"""Microbenchmarks for Conv3d and ConvTranspose3d operators."""
conv3d_fuzzed_configs_short = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV3D,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="Conv3D",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
    checksum=872,
)

conv3d_fuzzed_configs_long = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.CONV3D,
    fuzz_utils.Scale.MEDIUM,
    n=10,
    seed="Conv3D",
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["long"],
    checksum=1915,
)

class Conv3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.conv3d = nn.Conv3d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE).to(device=device)
        self.set_module_name('Conv3d')

    def forward(self):
        return self.conv3d(self.input)


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, C_OUT, KERNEL_SIZE, STRIDE, device):
        self.input = torch.rand(X_SIZE, device=device, requires_grad=self.auto_set())
        self.convtranspose3d = nn.ConvTranspose1d(
            X_SIZE[1], C_OUT, KERNEL_SIZE, stride=STRIDE).to(device=device)
        self.set_module_name('ConvTranspose3d')

    def forward(self):
        return self.convtranspose3d(self.input)

conv3d_configs = conv3d_fuzzed_configs_short + conv3d_fuzzed_configs_long
op_bench.generate_pt_test(conv3d_configs, Conv3dBenchmark)
op_bench.generate_pt_test(conv3d_configs, ConvTranspose3dBenchmark)
op_bench.generate_pt_gradient_test(conv3d_configs, Conv3dBenchmark)
op_bench.generate_pt_gradient_test(conv3d_configs, ConvTranspose3dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
