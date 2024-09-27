from utils.common import BenchmarkConfig

import torch

from . import FusedLinearCrossEntropyOperator, H, V
from .baseline import TorchLMHeadCE


class TorchLMHeadCECompiled(TorchLMHeadCE):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__(H, V, dtype, ignore_index)


class Operator(FusedLinearCrossEntropyOperator):
    variant = "Inductor"

    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(benchmark_config)
        self.operator = TorchLMHeadCECompiled(
            H=H, V=V, dtype=self.benchmark_config.dtype
        ).to(self.benchmark_config.device.value)
        self.operator = torch.compile(self.operator)
