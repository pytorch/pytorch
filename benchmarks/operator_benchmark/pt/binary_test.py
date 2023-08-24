import torch

import operator_benchmark as op_bench


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device, dtype=dtype),
            "in_two": torch.randn(in_two, device=device, dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


def copy(in1, in2):
    return in1.copy_(in2)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["copy_", copy],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [torch.int32, torch.bfloat16],
        "dtype_two": [torch.int32, torch.bfloat16],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32, torch.bfloat16],
    dtype_two=[torch.int8, torch.int32, torch.bfloat16],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


other_binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["atan2", torch.atan2],
        ["logaddexp", torch.logaddexp],
        ["logaddexp2", torch.logaddexp2],
        ["hypot", torch.hypot],
        ["igamma", torch.igamma],
        ["igammac", torch.igammac],
        ["xlogy", torch.xlogy],
        ["xlog1py", torch.special.xlog1py],
        ["zeta", torch.special.zeta],
        ["chebyshev_polynomial_t", torch.special.chebyshev_polynomial_t],
        ["chebyshev_polynomial_u", torch.special.chebyshev_polynomial_u],
        ["chebyshev_polynomial_v", torch.special.chebyshev_polynomial_v],
        ["chebyshev_polynomial_w", torch.special.chebyshev_polynomial_w],
        ["hermite_polynomial_h", torch.special.hermite_polynomial_h],
        ["hermite_polynomial_he", torch.special.hermite_polynomial_he],
        ["laguerre_polynomial_l", torch.special.laguerre_polynomial_l],
        ["laguerre_polynomial_p", torch.special.legendre_polynomial_p],
        [
            "shifted_chebyshev_polynomial_t",
            torch.special.shifted_chebyshev_polynomial_t,
        ],
        [
            "shifted_chebyshev_polynomial_u",
            torch.special.shifted_chebyshev_polynomial_u,
        ],
        [
            "shifted_chebyshev_polynomial_v",
            torch.special.shifted_chebyshev_polynomial_v,
        ],
        [
            "shifted_chebyshev_polynomial_w",
            torch.special.shifted_chebyshev_polynomial_w,
        ],
    ],
)

other_binary_ops_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.float32],
    tags=["long"],
)

other_binary_ops_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float32],
    },
    tags=["short"],
)


class SpecialBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device, dtype=dtype) * 10,
            "input_two": torch.randn(M, N, K, device=device, dtype=dtype) * 10,
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    other_binary_ops_list,
    other_binary_ops_long_configs + other_binary_ops_short_configs,
    SpecialBinaryOpBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
