import torch
import torch.nn as nn

import operator_benchmark as op_bench


"""
Microbenchmarks for the hardswish operators.
"""


# Configs for hardswish ops
hardswish_configs_short = op_bench.config_list(
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


hardswish_configs_long = op_bench.cross_product_configs(
    N=[8, 16], C=[3], H=[256, 512], W=[256, 512], device=["cpu"], tags=["long"]
)


hardswish_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardswish", nn.Hardswish],
    ],
)


class HardswishBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func):
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device)}
        self.op_func = op_func()

    def forward(self, input_one):
        return self.op_func(input_one)


op_bench.generate_pt_tests_from_op_list(
    hardswish_ops_list,
    hardswish_configs_short + hardswish_configs_long,
    HardswishBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
