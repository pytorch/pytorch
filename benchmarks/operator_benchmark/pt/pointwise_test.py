import torch

import operator_benchmark as op_bench

"""
Microbenchmarks for some pointwise operators.
"""

# Config for some pointwise ops
pointwise_configs = op_bench.config_list(
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
    },
    tags=["short"],
)


pointwise_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["addcmul", torch.addcmul], ["addcdiv", torch.addcdiv]],
)


class PointwiseBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, J, K, L, device, dtype, op_func):
        self.inputs = {
            "input_one": torch.rand(J, K, L, device=device, dtype=dtype),
            "input_two": torch.rand(J, K, L, device=device, dtype=dtype),
            "input_three": torch.rand(J, K, L, device=device, dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two, input_three):
        return self.op_func(input_one, input_two, input_three)


op_bench.generate_pt_tests_from_op_list(
    pointwise_ops_list, pointwise_configs, PointwiseBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
