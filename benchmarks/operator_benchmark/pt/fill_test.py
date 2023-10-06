import torch

from torch.testing._internal.common_device_type import get_all_device_types

import operator_benchmark as op_bench

"""Microbenchmark for Fill_ operator."""

fill_short_configs = op_bench.config_list(
    attr_names=["N"],
    attrs=[
        [1],
        [1024],
        [2048],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.int32],
    },
    tags=["short"],
)

fill_long_configs = op_bench.cross_product_configs(
    N=[10, 1000],
    device=get_all_device_types(),
    dtype=[
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.float,
        torch.double,
    ],
    tags=["long"],
)


class Fill_Benchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, device, dtype):
        self.inputs = {"input_one": torch.zeros(N, device=device).type(dtype)}
        self.set_module_name("fill_")

    def forward(self, input_one):
        return input_one.fill_(10)


op_bench.generate_pt_test(fill_short_configs + fill_long_configs, Fill_Benchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
