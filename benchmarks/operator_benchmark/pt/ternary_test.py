import operator_benchmark as op_bench
import torch


"""Microbenchmarks for ternary operators."""


ternary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["addcmul", torch.addcmul],
        ["addcdiv", torch.addcdiv],
    ],
)

ternary_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [1, 2],
        [32, 64],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16],
    },
    tags=["short"],
)

ternary_configs_long = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


class TernaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype, op_func):
        self.inputs = {
            "input_": torch.rand((M, N), device=device).to(dtype=dtype),
            "tensor1": torch.rand((M, N), device=device).to(dtype=dtype),
            "tensor2": torch.rand((M, N), device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, input_, tensor1, tensor2):
        return self.op_func(input_, tensor1, tensor2)


op_bench.generate_pt_tests_from_op_list(
    ternary_ops, ternary_configs_short + ternary_configs_long, TernaryOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
