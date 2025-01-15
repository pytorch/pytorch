import operator_benchmark as op_bench

import torch


"""Microbenchmarks for remainder operators."""


# Benchmark ops performance with broadcast
remainder_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["fmod", torch.fmod],
        ["remainder", torch.remainder],
    ],
)

remainder_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.int32, torch.float, torch.double],
    },
    tags=["short"],
)

remainder_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.int32, torch.float, torch.double],
    tags=["long"],
)


class RemainderOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype, op_func):
        self.dividend = torch.rand(M, N, K, device=device)
        self.dividend = (self.dividend * 1000 - 500).to(dtype=dtype)

        self.divisor = torch.rand(M, N, K, device=device)
        # +1 so we don't divide by zero
        self.divisor = (self.divisor * 40 + 1).to(dtype=dtype)

        self.inputs = {"dividend": self.dividend, "divisor": self.divisor}

        self.op_func = op_func

    def forward(self, dividend, divisor):
        return self.op_func(dividend, divisor)


op_bench.generate_pt_tests_from_op_list(
    remainder_ops_list,
    remainder_short_configs + remainder_long_configs,
    RemainderOpBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
