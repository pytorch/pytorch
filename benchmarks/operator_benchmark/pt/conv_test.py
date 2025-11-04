from pt import configs

import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""
Microbenchmarks for Conv1d and ConvTranspose1d operators.
"""


class Conv1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device):
        self.inputs = {
            "input": torch.rand(N, IC, L, device=device, requires_grad=self.auto_set())
        }
        self.conv1d = nn.Conv1d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("Conv1d")

    def forward(self, input):
        return self.conv1d(input)


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device):
        self.inputs = {"input": torch.rand(N, IC, L, device=device)}
        self.convtranspose1d = nn.ConvTranspose1d(IC, OC, kernel, stride=stride).to(
            device=device
        )
        self.set_module_name("ConvTranspose1d")

    def forward(self, input):
        return self.convtranspose1d(input)


op_bench.generate_pt_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long, Conv1dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    Conv1dBenchmark,
)


if not torch.backends.mkldnn.is_acl_available():
    # convtranpose1d crashes with ACL, see https://github.com/pytorch/pytorch/issues/165654
    op_bench.generate_pt_test(
        configs.convtranspose_1d_configs_short
        + configs.conv_1d_configs_short
        + configs.conv_1d_configs_long,
        ConvTranspose1dBenchmark,
    )


"""
Microbenchmarks for Conv2d, ConvTranspose2d, and Conv2dPointwise operators.
"""


class Conv2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.conv2d = nn.Conv2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("Conv2d")

    def forward(self, input):
        return self.conv2d(input)


class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.convtranspose2d = nn.ConvTranspose2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("ConvTranspose2d")

    def forward(self, input):
        return self.convtranspose2d(input)


class Conv2dPointwiseBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        # Use 1 as kernel for pointwise convolution
        self.conv2d = nn.Conv2d(IC, OC, 1, stride=stride, groups=G, padding=pad).to(
            device=device
        )
        self.set_module_name("Conv2dPointwise")

    def forward(self, input):
        return self.conv2d(input)


op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long, Conv2dBenchmark
)
op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long,
    ConvTranspose2dBenchmark,
)
op_bench.generate_pt_test(
    configs.conv_2d_pw_configs_short + configs.conv_2d_pw_configs_long,
    Conv2dPointwiseBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    Conv2dBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    ConvTranspose2dBenchmark,
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(
        configs.conv_2d_pw_configs_short + configs.conv_2d_pw_configs_long
    ),
    Conv2dPointwiseBenchmark,
)


"""
Microbenchmarks for Conv3d and ConvTranspose3d operators.
"""


class Conv3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device)}
        self.conv3d = nn.Conv3d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("Conv3d")

    def forward(self, input):
        return self.conv3d(input)


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device)}
        self.convtranspose3d = nn.ConvTranspose3d(IC, OC, kernel, stride=stride).to(
            device=device
        )
        self.set_module_name("ConvTranspose3d")

    def forward(self, input):
        return self.convtranspose3d(input)


op_bench.generate_pt_test(configs.conv_3d_configs_short, Conv3dBenchmark)
op_bench.generate_pt_test(configs.conv_3d_configs_short, ConvTranspose3dBenchmark)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_3d_configs_long), Conv3dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_3d_configs_long), ConvTranspose3dBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
