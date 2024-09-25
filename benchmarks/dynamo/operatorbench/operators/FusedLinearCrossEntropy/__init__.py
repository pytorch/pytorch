from typing import Any, List, Callable

from utils.common import BenchmarkConfig, Phase

import torch

from .. import BaseOperator


H = 4096
V = 128256
# Each file defines an operator variant
valid_operator_files = ["baseline.py", "custom.py", "inductor.py"]


# Reference: https://github.com/linkedin/Liger-Kernel/blob/\
# 3d0653b035222cbb845435a1994854e4fd219107/benchmark/scripts/benchmark_fused_linear_cross_entropy.py


class FusedLinearCrossEntropyOperator(BaseOperator):
    # The base operator name
    name = "FusedLinearCrossEntropy"
    # The variant placeholder. No need to set in the base operator class
    variant = None

    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(benchmark_config)
        self.forward_output = None

    @classmethod
    def generate_inputs(cls, benchmark_config: BenchmarkConfig):
        # May need OOM check
        for BT in [2**i for i in range(12, 16)]:
            _input = torch.randn(
                BT,
                H,
                requires_grad=True,
                dtype=benchmark_config.dtype,
                device=benchmark_config.device.value,
            )
            target = torch.randint(
                V, (BT, 1), dtype=torch.long, device=benchmark_config.device.value
            ).squeeze(1)
            # This operator needs two inputs
            cls.example_inputs_list.append((_input, target))

    def forward(self, input: Any):
        return self.operator(input)

    # backward doesn't need inputs, but we need to pass it to match the interface
    def backward(self, input: Any):
        assert self.forward_output is not None
        return self.forward_output.backward(retain_graph=True)

    def full(self, input: Any):
        y = self.forward(input)
        y.backward()
        return y

    def prepare_input_and_functions(self, input: Any, phase: Phase):
        if phase == Phase.BACKWARD:
            self.forward_output = self.forward(input)
        return input
