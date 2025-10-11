import operator_benchmark as op_bench

import torch


"""Microbenchmarks for batched operators."""

# binary ops (two inputs in shape of batches)
batched_binary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["bmm", torch.bmm],
    ],
)

batched_binary_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [2, 1, 8, 2],
        [128, 64, 32, 64],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16],
    },
    tags=["short"],
)

batched_binary_configs_long = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    N=[256, 1024],
    K=[64, 128],
    device=["cuda"],
    dtype=[torch.float32, torch.bfloat16, torch.float16],
    tags=["long"],
)


class BatchedBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype, op_func):
        self.inputs = {
            "batch1": torch.rand(
                (B, M, N), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (B, N, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
        }
        self.op_func = op_func

    def forward(self, batch1, batch2):
        return self.op_func(batch1, batch2)


op_bench.generate_pt_tests_from_op_list(
    batched_binary_ops,
    batched_binary_configs_short + batched_binary_configs_long,
    BatchedBinaryOpBenchmark,
)
op_bench.generate_pt_gradient_tests_from_op_list(
    batched_binary_ops,
    batched_binary_configs_long,
    BatchedBinaryOpBenchmark,
)


# batched ternary ops
batched_ternary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["baddbmm", torch.baddbmm]],
)


class BatchedTernaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype, op_func):
        self.inputs = {
            "input_": torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch1": torch.rand(
                (B, M, N), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (B, N, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
        }
        self.op_func = op_func

    def forward(self, input_, batch1, batch2):
        return self.op_func(input_, batch1, batch2)


op_bench.generate_pt_tests_from_op_list(
    batched_ternary_ops,
    batched_binary_configs_short + batched_binary_configs_long,
    BatchedTernaryOpBenchmark,
)
op_bench.generate_pt_gradient_tests_from_op_list(
    batched_ternary_ops,
    batched_binary_configs_long,
    BatchedTernaryOpBenchmark,
)


# TODO: does it automatically register new scripts?

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
