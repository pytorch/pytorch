import torch
import torch.nn as nn

import operator_benchmark as op_bench


"""
Microbenchmarks for some activation functions.
"""


configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float32, torch.bfloat16]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=["cpu"],
    dtype=[torch.float32, torch.bfloat16],
    tags=["long"],
)


ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardswish", nn.Hardswish],
        ["Hardshrink", nn.Hardshrink],
        ["Softshrink", nn.Softshrink],
        ["Hardtanh", nn.Hardtanh],
        ["LogSigmoid", nn.LogSigmoid],
        ["Hardsigmoid", nn.Hardsigmoid],
        ["ELU", nn.ELU],
        ["GELU", nn.GELU],
        ["GLU", nn.GLU],
        ["SiLU", nn.SiLU],
        ["Softplus", nn.Softplus],
        ["Mish", nn.Mish],
    ],
)


inplace_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardtanh_", nn.Hardtanh],
        ["Hardswish_", nn.Hardswish],
        ["Hardsigmoid_", nn.Hardsigmoid],
        ["SiLU_", nn.SiLU],
        ["ELU_", nn.ELU],
        ["Mish_", nn.Mish],
    ],
)


class ActivationBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, dtype, op_func):
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device, dtype=dtype)}
        self.op_func = op_func()

    def forward(self, input_one):
        return self.op_func(input_one)


op_bench.generate_pt_tests_from_op_list(
    ops_list, configs_short + configs_long, ActivationBenchmark
)


class InplaceActivationBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, dtype, op_func):
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device, dtype=dtype)}
        self.op_func = op_func(inplace=True)

    def forward(self, input_one):
        return self.op_func(input_one)


op_bench.generate_pt_tests_from_op_list(
    inplace_ops_list, configs_short + configs_long, InplaceActivationBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
