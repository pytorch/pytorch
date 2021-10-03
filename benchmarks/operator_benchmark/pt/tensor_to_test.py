import operator_benchmark as op_bench
import torch

tensor_conversion_short_configs = op_bench.cross_product_configs(
    M=(8, 16, 32,),
    N=(16, 64, 128,),
    device=['cpu', 'cuda'],
    tags=['short'],
)

tensor_conversion_long_configs = op_bench.cross_product_configs(
    M=(64, 128, 256, 512,),
    N=(256, 512, 1024, 2048,),
    device=['cpu', 'cuda'],
    tags=['long'],
)

class FloatToHalfTensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device, requires_grad=False, dtype=torch.float)
        }

    def forward(self, input):
        return input.to(torch.half)

class HalfToFloatTensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device, requires_grad=False, dtype=torch.half)
        }

    def forward(self, input):
        return input.to(torch.float)


op_bench.generate_pt_test(tensor_conversion_short_configs, FloatToHalfTensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_long_configs, FloatToHalfTensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_short_configs, HalfToFloatTensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_long_configs, HalfToFloatTensorConversionBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
