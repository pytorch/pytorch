import torch
import torch.nn.functional as F

import operator_benchmark as op_bench


"""Microbenchmarks for instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    tags=["short"],
    dtype=[torch.float32, torch.bfloat16],
)


class InstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, dtype):
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims, dtype=dtype) - 0.5) * 256,
            "weight": torch.rand(num_channels, dtype=dtype),
            "bias": torch.rand(num_channels, dtype=dtype),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        return F.instance_norm(input, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(instancenorm_configs_short, InstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
