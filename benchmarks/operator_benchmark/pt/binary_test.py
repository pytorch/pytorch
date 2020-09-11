import benchmark_fuzz_utils as fuzz_utils
import operator_benchmark as op_bench
import torch

binary_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['add', torch.add],
        ['mul', torch.mul],
        ['div', getattr(torch, "true_divide", torch.div)],
    ],
)

binary_short_float_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BINARY,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed='Add',
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },
    tags=['short'],
    checksum=1431,
)

binary_long_float_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BINARY,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGE,
    n=10,
    seed='Add',
    tags=['long'],
    checksum=(8505, 11877560),
)

class BinaryFloatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device, op_func):
        self.input_one = torch.rand(X_SIZE, device=device, dtype=torch.float, requires_grad=self.auto_set())
        self.input_two = torch.rand(Y_SIZE, device=device, dtype=torch.float, requires_grad=self.auto_set())
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one, self.input_two)

op_bench.generate_pt_tests_from_op_list(
    binary_ops_list,
    binary_short_float_configs + binary_long_float_configs,
    BinaryFloatBenchmark)

op_bench.generate_pt_gradient_tests_from_op_list(
    binary_ops_list,
    binary_short_float_configs + binary_long_float_configs,
    BinaryFloatBenchmark)


binary_short_int_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BINARY,
    fuzz_utils.Scale.SMALL,
    n=10,
    seed="Add",
    cross_product_configs={
        'device': ['cpu', 'cuda'],
        'dtypes': [
            (torch.int32, torch.int32),
            (torch.int8, torch.int8),
            (torch.int32, torch.int8),  # Test type promotion.
        ],
    },
    tags=["short"],
    checksum=1431,
)

binary_long_int_configs = fuzz_utils.make_fuzzed_config(
    fuzz_utils.Fuzzers.BINARY,
    fuzz_utils.CPU_MEDIUM_CUDA_LARGE,
    n=10,
    seed="Add",
    cross_product_configs={
        'dtypes': [
            (torch.int32, torch.int32),
            (torch.int8, torch.int8),
            (torch.int32, torch.int8),  # Test type promotion.
        ],
    },
    tags=["long"],
    checksum=(8505, 11877560),
)

class BinaryIntBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, X_SIZE, Y_SIZE, device, dtypes, op_func):
        self.input_one = torch.rand(X_SIZE, device=device).to(dtype=dtypes[0])
        self.input_two = torch.rand(Y_SIZE, device=device).to(dtype=dtypes[1])
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one, self.input_two)

op_bench.generate_pt_tests_from_op_list(
    binary_ops_list,
    binary_short_int_configs + binary_long_int_configs,
    BinaryIntBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
