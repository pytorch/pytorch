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

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
