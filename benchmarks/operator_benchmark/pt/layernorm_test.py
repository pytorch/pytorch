import operator_benchmark as op_bench

import torch
import torch.nn.functional as F


"""Microbenchmarks for layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    # affine selects which of weight/bias are passed; None args hit the
    # vectorized null-affine paths in the CPU LayerNormSecondPass kernel.
    affine=("both", "weight_only", "bias_only", "none"),
    tags=["short"],
)


class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, affine):
        input = (torch.rand(*dims) - 0.5) * 256
        normalized_shape = input.size()[1:]
        weight = torch.rand(*normalized_shape, dtype=torch.float)
        bias = torch.rand(*normalized_shape, dtype=torch.float)
        self.inputs = {
            "input": input,
            "weight": weight if affine in ("both", "weight_only") else None,
            "bias": bias if affine in ("both", "bias_only") else None,
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        return F.layer_norm(input, input.size()[1:], weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(layernorm_configs_short, LayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
