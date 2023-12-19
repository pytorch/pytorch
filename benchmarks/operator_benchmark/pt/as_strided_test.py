from typing import List

import torch

import operator_benchmark as op_bench


"""Microbenchmarks for as_strided operator"""


# Configs for PT as_strided operator
as_strided_configs_short = op_bench.config_list(
    attr_names=["M", "N", "size", "stride", "storage_offset"],
    attrs=[
        [8, 8, (2, 2), (1, 1), 0],
        [256, 256, (32, 32), (1, 1), 0],
        [512, 512, (64, 64), (2, 2), 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

as_strided_configs_long = op_bench.cross_product_configs(
    M=[512],
    N=[1024],
    size=[(16, 16), (128, 128)],
    stride=[(1, 1)],
    storage_offset=[0, 1],
    device=["cpu", "cuda"],
    tags=["long"],
)


class As_stridedBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, size, stride, storage_offset, device):
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "size": size,
            "stride": stride,
            "storage_offset": storage_offset,
        }
        self.set_module_name("as_strided")

    def forward(
        self, input_one, size: List[int], stride: List[int], storage_offset: int
    ):
        return torch.as_strided(input_one, size, stride, storage_offset)


op_bench.generate_pt_test(
    as_strided_configs_short + as_strided_configs_long, As_stridedBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
