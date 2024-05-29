import math

import torch

import operator_benchmark as op_bench


"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""

# Configs for PT torch.nan_to_num / nan_to_num_ operators

nan_to_num_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["nan_to_num", torch.nan_to_num],
        ["nan_to_num_", torch.nan_to_num_],
    ],
)

nan_to_num_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128],
    N=range(32, 128, 32),
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["long"],
)


nan_to_num_short_configs = op_bench.cross_product_configs(
    M=[16, 64],
    N=[64, 64],
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["short"],
)


class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, replace_inf, op_func):
        input = torch.randn(M, N, dtype=dtype)
        input[0][0] = float("nan")
        self.inputs = {"input": input, "replace_inf": replace_inf}
        self.op_func = op_func
        self.set_module_name("nan_to_num")

    def forward(self, input, replace_inf: bool):
        # compare inplace
        if replace_inf:
            return self.op_func(input, nan=1.0)
        else:
            return self.op_func(input, nan=1.0, posinf=math.inf, neginf=-math.inf)


op_bench.generate_pt_tests_from_op_list(
    nan_to_num_ops_list,
    nan_to_num_long_configs + nan_to_num_short_configs,
    ReplaceNaNBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
