import operator_benchmark as op_bench

import torch


"""Microbenchmarks for MatMul operator"""

# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dtype": [torch.float]},
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                M, N, device=device, dtype=dtype, requires_grad=self.auto_set()
            )
            if trans_a
            else torch.rand(N, M, device=device, dtype=dtype)
            .t()
            .contiguous()
            .clone()
            .detach()
            .requires_grad_(self.auto_set()),
            "input_two": torch.rand(
                N, K, device=device, dtype=dtype, requires_grad=self.auto_set()
            )
            if trans_b
            else torch.rand(K, N, device=device, dtype=dtype)
            .t()
            .contiguous()
            .clone()
            .detach()
            .requires_grad_(self.auto_set()),
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)
op_bench.generate_pt_gradient_test(mm_long_configs + mm_short_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
