import numpy as np
import torch

import operator_benchmark as op_bench

"""
Microbenchmarks for the polar operator.
"""

# Config for polar op
polar_config = op_bench.config_list(
    attr_names=["L", "M", "N"],
    attrs=[
        [3, 256, 256],
        [3, 64, 64],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float32, torch.float64]},
    tags=["short"],
)


polar_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["polar", torch.polar],
    ],
)


class PolarBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, L, M, N, device, dtype, op_func):
        self.inputs = {
            "input_one": torch.rand(L, M, N, device=device, dtype=dtype),
            "input_two": torch.tensor(
                [5 * np.pi / 4], device=device, dtype=dtype
            ).expand(L, M, N),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(polar_op_list, polar_config, PolarBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
