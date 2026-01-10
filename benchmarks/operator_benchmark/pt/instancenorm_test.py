import operator_benchmark as op_bench
import torch
import torch.nn.functional as F


"""Microbenchmarks for instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    tags=["short"],
)


class InstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims):
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims) - 0.5) * 256,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        return F.instance_norm(input, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(instancenorm_configs_short, InstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
