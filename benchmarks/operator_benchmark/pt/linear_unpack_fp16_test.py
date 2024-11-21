import operator_benchmark as op_bench

import torch


"""Microbenchmarks for linear_unpack_fp16_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT linear_unpack_fp16 operator
linear_unpack_fp16_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"]
)

linear_unpack_fp16_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)


class LinearUnpackFP16Benchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        # input to unpack operator must be what the output is for prepack operator
        self.inputs = {
            "input_one": torch.ops.quantized.linear_prepack_fp16(
                torch.rand(
                    M, N, K, device=device, requires_grad=False, dtype=torch.float32
                )
            )
        }
        self.set_module_name("linear_unpack_fp16")

    def forward(self, input_one):
        return torch.ops.quantized.linear_unpack_fp16(input_one)


# The generated test names based on linear_unpack_fp16_short_configs will be in the following pattern:
# linear_unpack_fp16_M8_N16_K32_devicecpu

op_bench.generate_pt_test(
    linear_unpack_fp16_long_configs + linear_unpack_fp16_short_configs,
    LinearUnpackFP16Benchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
