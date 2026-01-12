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

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for Conv1d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv1d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × L
        input_elements = input_tensor.numel()
        # Weight: OC × IC × kernel
        weight_elements = self.conv1d.weight.numel()
        # Output: N × OC × L_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose1dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device):
        self.inputs = {"input": torch.rand(N, IC, L, device=device)}
        self.convtranspose1d = nn.ConvTranspose1d(IC, OC, kernel, stride=stride).to(
            device=device
        )
        self.set_module_name("ConvTranspose1d")

    def forward(self, input):
        return self.convtranspose1d(input)

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for ConvTranspose1d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose1d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × L
        input_elements = input_tensor.numel()
        # Weight: IC × OC × kernel
        weight_elements = self.convtranspose1d.weight.numel()
        # Output: N × OC × L_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


op_bench.generate_pt_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long, Conv1dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.remove_cpu(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    Conv1dBenchmark,
)

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

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for Conv2d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × (IC/G) × kernel × kernel
        weight_elements = self.conv2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose2dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.convtranspose2d = nn.ConvTranspose2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("ConvTranspose2d")

    def forward(self, input):
        return self.convtranspose2d(input)

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for ConvTranspose2d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: IC × (OC/G) × kernel × kernel
        weight_elements = self.convtranspose2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


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

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for Conv2dPointwise: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv2d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × (IC/G) × 1 × 1
        weight_elements = self.conv2d.weight.numel()
        # Output: N × OC × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


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

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for Conv3d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.conv3d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × D × H × W
        input_elements = input_tensor.numel()
        # Weight: OC × IC × kernel × kernel × kernel
        weight_elements = self.conv3d.weight.numel()
        # Output: N × OC × D_out × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


class ConvTranspose3dBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device)}
        self.convtranspose3d = nn.ConvTranspose3d(IC, OC, kernel, stride=stride).to(
            device=device
        )
        self.set_module_name("ConvTranspose3d")

    def forward(self, input):
        return self.convtranspose3d(input)

    def get_memory_traffic_bytes(self):
        """Calculate memory traffic for ConvTranspose3d: read(input + weight) + write(output)"""
        input_tensor = self.inputs["input"]
        # Run forward to get output shape
        with torch.no_grad():
            output = self.convtranspose3d(input_tensor)

        bytes_per_element = input_tensor.element_size()
        # Input: N × IC × D × H × W
        input_elements = input_tensor.numel()
        # Weight: IC × OC × kernel × kernel × kernel
        weight_elements = self.convtranspose3d.weight.numel()
        # Output: N × OC × D_out × H_out × W_out
        output_elements = output.numel()

        total_elements = input_elements + weight_elements + output_elements
        return total_elements * bytes_per_element


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
