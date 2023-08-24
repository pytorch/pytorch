import torch

import operator_benchmark as op_bench

"""
Microbenchmarks for some reduction operators.
"""

# Config for some reduction ops
reduction_configs = op_bench.config_list(
    attr_names=[
        "J",
        "K",
        "L",
    ],
    attrs=[
        [3, 8, 32],
        [1, 1, 64],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.bfloat16, torch.float32, torch.float64],
        "dim": [0, 1, 2],
    },
    tags=["short"],
)


reduction_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["cumprod", torch.cumprod],
        ["cumsum", torch.cumsum],
        ["logcumsumexp", torch.logcumsumexp],
    ],
)


class ReductionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, J, K, L, device, dtype, op_func, dim):
        self.inputs = {
            "input_one": torch.rand(J, K, L, device=device, dtype=dtype),
        }
        self.op_func = op_func
        self.dim = dim

    def forward(self, input_one):
        return self.op_func(input_one, dim=self.dim)


op_bench.generate_pt_tests_from_op_list(
    reduction_ops_list, reduction_configs, ReductionBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
