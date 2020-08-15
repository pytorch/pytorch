import operator_benchmark as op_bench
import torch


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['add', torch.add],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=['in_one', 'in_two'],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        'device': ['cpu'],
        'dtype': [torch.float],
    },
    tags=["short"]
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.in_one = torch.randn(in_one, device=device).to(dtype=dtype)
        self.in_two = torch.randn(in_two, device=device).to(dtype=dtype)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.in_one, self.in_two)


op_bench.generate_pt_tests_from_op_list(binary_ops_bcast_list,
                                        binary_configs_broadcast,
                                        BinaryOpBcastBenchmark)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['add', torch.add],
        ['copy_', lambda in1, in2: in1.copy_(in2)],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=['M', 'N', 'K'],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
        'dtype_one' : [torch.int32],
        'dtype_two' : [torch.int32],
    },
    tags=['short'],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=['cpu', 'cuda'],
    dtype_one=[torch.int8, torch.int32],
    dtype_two=[torch.int8, torch.int32],
    tags=['long']
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.input_one = torch.randn(M, N, K, device=device).to(dtype=dtype_one)
        self.input_two = torch.randn(M, N, K, device=device).to(dtype=dtype_two)
        self.op_func = op_func

    def forward(self):
        return self.op_func(self.input_one, self.input_two)


op_bench.generate_pt_tests_from_op_list(binary_ops_list,
                                        binary_short_configs + binary_long_configs,
                                        BinaryOpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
