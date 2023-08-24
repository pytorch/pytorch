import torch

import operator_benchmark as op_bench

"""Microbenchmarks for MatMul operator"""

# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float32, torch.bfloat16],
    },
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
    M=[32],
    N=[512, 128],
    K=[64],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cpu", "cuda"],
    dtype=[torch.float32, torch.bfloat16],
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b, device, dtype):
        self.inputs = {
            "input_one": torch.rand(M, N, device=device, dtype=dtype)
            if trans_a
            else torch.rand(N, M, device=device, dtype=dtype).t(),
            "input_two": torch.rand(N, K, device=device, dtype=dtype)
            if trans_b
            else torch.rand(K, N, device=device, dtype=dtype).t(),
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
