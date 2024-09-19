from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)
from utils.common import BenchmarkConfig

import torch

from . import FusedLinearCrossEntropyOperator, H, V


# Reference: https://github.com/linkedin/Liger-Kernel/blob/\
# 3d0653b035222cbb845435a1994854e4fd219107/benchmark/scripts/benchmark_fused_linear_cross_entropy.py#L40
class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


class Operator(FusedLinearCrossEntropyOperator):
    variant = "Liger"

    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(benchmark_config)
        self.operator = LigerLMHeadCE(H=H, V=V, dtype=self.benchmark_config.dtype).to(
            self.benchmark_config.device.value
        )
