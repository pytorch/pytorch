import operator_benchmark as op_bench

import torch


"""
Microbenchmarks for batch matrix mult with einsum and torch.bmm.
"""

batch_mm_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [4, 5, 3, 2],
        [32, 25, 20, 30],
        [128, 100, 120, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

batch_mm_configs_long = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [128, 256, 128, 256],
        [512, 1024, 1024, 512],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

batch_mm_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_bmm", torch.einsum],
        ["bmm", torch.bmm],
    ],
)


class BatchMatrixMultBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, op_func):
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, N, K, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bjk->bik", input_one, input_two)
        else:
            return torch.bmm(input_one, input_two)


"""
Microbenchmarks for element-wise matrix mult with einsum and torch.mul.
"""

batch_elementwise_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N"],
    attrs=[
        [4, 5, 3],
        [32, 25, 20],
        [100, 90, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


batch_elementwise_configs_long = op_bench.cross_product_configs(
    B=[128, 512, 1024],
    M=[128, 512, 1024],
    N=[128, 512, 1024],
    device=["cpu", "cuda"],
    tags=["long"],
)

batch_elementwise_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_elementwise", torch.einsum],
        ["mul", torch.mul],
    ],
)


class BatchElementWiseBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, device, op_func):
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, M, N, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bij->bij", input_one, input_two)
        else:
            return torch.mul(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    batch_mm_op_list,
    batch_mm_configs_short + batch_mm_configs_long,
    BatchMatrixMultBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    batch_elementwise_op_list,
    batch_elementwise_configs_short + batch_elementwise_configs_long,
    BatchElementWiseBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
