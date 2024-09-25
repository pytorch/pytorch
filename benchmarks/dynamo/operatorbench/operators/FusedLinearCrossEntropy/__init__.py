from typing import Optional

from utils.common import BenchmarkConfig

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

    def forward(self, inputs):
        return self.operator(inputs)

    def backward(self, inputs, forward_output):
        return forward_output.backward(retain_graph=True)

    def full(self, input):
        y = self.forward(input)
        y.backward()
        return y

    # single run with a specific input
    def single_run(self, fn, inputs):
        fn(inputs)

    def prepare_input_and_functions(self, input):
        self.forward_output = self.forward(input)
        return input
