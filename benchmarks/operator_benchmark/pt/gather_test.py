import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for gather operator."""

# An example input from this configuration is M=4, N=4, dim=0.
gather_configs_short = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


gather_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"]
)


class GatherBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dim, device):
        min_val = M if dim == 0 else N
        numpy.random.seed((1 << 32) - 1)
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, min_val, (M, N)), device=device
            ),
        }
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        return torch.gather(input_one, dim, index)


op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
