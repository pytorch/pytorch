import operator_benchmark as op_bench

import torch


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
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
        "dtype": [torch.float, torch.bfloat16, torch.float64],
    },
    tags=["short"],
)

# Inner-dim broadcast on CUDA: bias adds [N] + [M, N] and a column broadcast
# [B, R, 1] + [B, R, N], at shapes from a recommendation model.
binary_configs_inner_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[8192], [17750, 8192]],
        [[3072], [17750, 3072]],
        [[1536], [17750, 1536]],
        [[8192], [28160, 8192]],
        [[71, 250, 1], [71, 250, 8192]],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16],
    },
    tags=["inner_broadcast"],
)

# Inner-dim broadcast that stays N-D after coalescing, at a recommendation
# model's shape.
binary_configs_inner_broadcast_nd = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[4096, 4, 4, 1], [4096, 1, 4, 1024]],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bfloat16],
    },
    tags=["inner_broadcast_nd"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list,
    binary_configs_broadcast
    + binary_configs_inner_broadcast
    + binary_configs_inner_broadcast_nd,
    BinaryOpBcastBenchmark,
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
        ["asr", torch.bitwise_right_shift],
        ["lsl", torch.bitwise_left_shift],
        ["xor", torch.bitwise_xor],
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
        "dtype_one": [torch.int32, torch.uint8],
        "dtype_two": [torch.int32, torch.uint8],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32, torch.uint8],
    dtype_two=[torch.int8, torch.int32, torch.uint8],
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


######
# Benchmark ops performance for boolean dtype
######


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.bool],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.bernoulli(0.5 * torch.ones(in_one, device=device)).to(
                dtype=dtype
            ),
            "in_two": torch.bernoulli(0.5 * torch.ones(in_two, device=device)).to(
                dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)

# Row mask broadcast along the inner dim on CUDA, at a recommendation model's shape.
binary_ops_bool_inner_broadcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["bitwise_or", torch.bitwise_or]],
)

binary_configs_bool_inner_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[28160, 8192], [28160, 1]],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bool],
    },
    tags=["inner_broadcast"],
)

op_bench.generate_pt_tests_from_op_list(
    binary_ops_bool_inner_broadcast_list,
    binary_configs_bool_inner_broadcast,
    BinaryOpBcastBenchmark,
)

# Outer-product broadcast of bool masks on CUDA (rows of 5000, a multiple of 8
# but not 16 bytes), at a recommendation model's shapes.
binary_ops_bool_outer_broadcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
)

binary_configs_bool_outer_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[17, 5000, 1], [17, 1, 5000]],
        [[17, 1, 5000, 5000], [17, 1, 1, 5000]],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "dtype": [torch.bool],
    },
    tags=["inner_broadcast_nd"],
)

op_bench.generate_pt_tests_from_op_list(
    binary_ops_bool_outer_broadcast_list,
    binary_configs_bool_outer_broadcast,
    BinaryOpBcastBenchmark,
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
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
        "dtype_one": [torch.bool],
        "dtype_two": [torch.bool],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.bool, torch.bool],
    dtype_two=[torch.bool, torch.bool],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_one
            ),
            "input_two": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_two
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
