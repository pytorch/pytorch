import operator_benchmark as op_bench

import torch


tensor_conversion_short_configs = op_bench.cross_product_configs(
    M=[32],
    N=[128],
    device=["cpu", "cuda"],
    dtype_one=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    dtype_two=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    tags=["short"],
)

tensor_conversion_long_configs = op_bench.cross_product_configs(
    M=[1024],
    N=[1024],
    device=["cpu", "cuda"],
    dtype_one=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    dtype_two=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    tags=["long"],
)

# bool -> uint8 on CUDA at a recommendation model's mask shapes, plus bool to
# other dtypes at the largest shape.
tensor_conversion_from_bool_configs = op_bench.config_list(
    attr_names=["M", "N", "dtype_one", "dtype_two"],
    attrs=[
        [17750, 8192, torch.bool, torch.uint8],
        [15750, 8192, torch.bool, torch.uint8],
        [12500, 8192, torch.bool, torch.uint8],
        [7500, 8192, torch.bool, torch.uint8],
        [17750, 8192, torch.bool, torch.int8],
        [17750, 8192, torch.bool, torch.int32],
        [17750, 8192, torch.bool, torch.half],
        [17750, 8192, torch.bool, torch.bfloat16],
        [17750, 8192, torch.bool, torch.float],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["from_bool"],
)

# Converting copies on CUDA at recommendation-model shapes, including small
# launch-bound ones.
tensor_conversion_converting_configs = op_bench.config_list(
    attr_names=["M", "N", "dtype_one", "dtype_two"],
    attrs=[
        [1, 7284636, torch.int64, torch.int32],
        [1, 63457, torch.float, torch.int64],
        [10916, 208, torch.half, torch.bfloat16],
        [6814, 1024, torch.bfloat16, torch.half],
        [71, 64, torch.int8, torch.bfloat16],
        [1, 4320, torch.int64, torch.float],
    ],
    cross_product_configs={"device": ["cuda"]},
    tags=["converting"],
)


class TensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype_one, dtype_two, device):
        self.inputs = {
            "input": torch.rand(
                M, N, device=device, requires_grad=False, dtype=torch.float
            ).to(dtype=dtype_one)
        }
        self.dtype_one = dtype_one
        self.dtype_two = dtype_two

    def forward(self, input):
        return input.to(dtype=self.dtype_two)


op_bench.generate_pt_test(tensor_conversion_short_configs, TensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_long_configs, TensorConversionBenchmark)
op_bench.generate_pt_test(
    tensor_conversion_from_bool_configs + tensor_conversion_converting_configs,
    TensorConversionBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
