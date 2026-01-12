from pt import configs

import operator_benchmark as op_bench
import torch
import torch.nn as nn


"""Microbenchmarks for Linear operator."""


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, IN, OUT, device):
        self.inputs = {"input_one": torch.rand(N, IN, device=device)}
        self.linear = nn.Linear(IN, OUT).to(device=device)
        self.set_module_name("linear")

    def forward(self, input_one):
        return self.linear(input_one)


op_bench.generate_pt_test(
    configs.linear_configs_short + configs.linear_configs_long, LinearBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
