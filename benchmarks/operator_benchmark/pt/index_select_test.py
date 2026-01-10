import numpy

import operator_benchmark as op_bench
import torch


"""Microbenchmarks for index_select operator."""

# An example input from this configuration is M=4, N=4, dim=0.
index_select_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[
        [8, 8, 1, 1],
        [256, 512, 1, 1],
        [512, 512, 1, 1],
        [8, 8, 2, 1],
        [256, 512, 2, 1],
        [512, 512, 2, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


index_select_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    dim=[1],
    device=["cpu", "cuda"],
    tags=["long"],
)


class IndexSelectBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, device):
        max_val = N
        numpy.random.seed((1 << 32) - 1)
        index_dim = numpy.random.randint(0, N)
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, max_val, index_dim), device=device
            ),
        }
        self.set_module_name("index_select")

    def forward(self, input_one, dim, index):
        return torch.index_select(input_one, dim, index)


op_bench.generate_pt_test(
    index_select_configs_short + index_select_configs_long, IndexSelectBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
