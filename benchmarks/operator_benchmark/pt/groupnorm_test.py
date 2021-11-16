
import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for groupnorm operator."""

groupnorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    num_groups=(2, 4),
    tags=["short"],
)


class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, num_groups):
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims) - 0.5) * 256,
            "num_groups": num_groups,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5
        }

    def forward(self, input, num_groups: int, weight, bias, eps: float):
        return F.group_norm(
            input, num_groups, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(groupnorm_configs_short, GroupNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
