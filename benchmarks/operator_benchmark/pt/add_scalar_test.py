import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.add with a scalar rhs."""


add_scalar_configs = op_bench.config_list(
    attr_names=["N", "alpha"],
    attrs=[
        [1, 1],
        [1, 3],
        [16, 1],
        [16, 3],
        [256, 1],
        [256, 3],
        [4096, 1],
        [4096, 3],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float],
    },
    tags=["short"],
)


class AddScalarBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, alpha, device, dtype):
        self.inputs = {
            "input": torch.rand(N, device=device, dtype=dtype),
            "other": 1.0,
            "alpha": alpha,
        }
        self.set_module_name("add_scalar")

    def forward(self, input, other, alpha):
        return torch.add(input, other, alpha=alpha)


op_bench.generate_pt_test(add_scalar_configs, AddScalarBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
