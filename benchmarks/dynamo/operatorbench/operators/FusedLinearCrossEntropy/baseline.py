import torch
from . import FusedLinearCrossEntropyOperator
from utils.common import BenchmarkConfig
from . import H, V

# Reference: https://github.com/linkedin/Liger-Kernel/blob/3d0653b035222cbb845435a1994854e4fd219107/benchmark/scripts/benchmark_fused_linear_cross_entropy.py#L17
class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class Operator(FusedLinearCrossEntropyOperator):
    variant = "Baseline"
    

    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(benchmark_config)
        self.operator = TorchLMHeadCE(H=H, V=V, dtype=self.benchmark_config.dtype).to(self.benchmark_config.device.value)


