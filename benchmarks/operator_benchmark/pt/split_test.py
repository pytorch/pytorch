import torch

import operator_benchmark as op_bench


"""Microbenchmarks for Split operator"""


# Configs for PT Split operator
split_configs_short = op_bench.config_list(
    attr_names=["M", "N", "parts"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

split_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], parts=[2, 4], device=["cpu", "cuda"], tags=["long"]
)


class SplitBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, parts, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device),
            "split_size": int(M * N / parts),
        }
        self.set_module_name("split")

    def forward(self, input, split_size: int):
        return torch.split(input, split_size)


op_bench.generate_pt_test(split_configs_short + split_configs_long, SplitBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
