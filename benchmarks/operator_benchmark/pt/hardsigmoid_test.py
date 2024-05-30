import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""
Microbenchmarks for the hardsigmoid operator.
"""


# Configs for hardsigmoid ops
hardsigmoid_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)


hardsigmoid_configs_long = op_bench.cross_product_configs(
    N=[8, 16], C=[3], H=[256, 512], W=[256, 512], device=["cpu"], tags=["long"]
)


hardsigmoid_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardsigmoid", nn.Hardsigmoid],
    ],
)


class HardsigmoidBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func):
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device)}
        self.op_func = op_func()

    def forward(self, input_one):
        return self.op_func(input_one)


op_bench.generate_pt_tests_from_op_list(
    hardsigmoid_ops_list,
    hardsigmoid_configs_short + hardsigmoid_configs_long,
    HardsigmoidBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
