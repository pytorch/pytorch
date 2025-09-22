"""Scalar multiply operator implementation."""

import random
from typing import List
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec


class ScalarMultiplyOperator(Operator):
    """Operator for multiplying two scalars."""

    def __init__(self):
        super().__init__("scalar_multiply")

    def can_produce(self, output_spec: Spec) -> bool:
        """Scalar multiply can only produce scalars."""
        return isinstance(output_spec, ScalarSpec)

    def supports_variable_inputs(self) -> bool:
        """Scalar multiply operator does not support variable number of inputs."""
        return False

    def decompose(self, output_spec: Spec, num_inputs: int = 2) -> List[Spec]:
        """Decompose scalar into input scalars for multiplication with type promotion."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError("ScalarMultiplyOperator can only produce ScalarSpec outputs")

        # Type promotion rules for scalars
        import torch
        promotion_table = {
            torch.float32: [
                (torch.float32, torch.float32),
                (torch.float16, torch.float32),
                (torch.float32, torch.float16),
                (torch.int32, torch.float32),
                (torch.float32, torch.int32),
            ],
            torch.float64: [
                (torch.float64, torch.float64),
                (torch.float32, torch.float64),
                (torch.float64, torch.float32),
            ],
            torch.int32: [
                (torch.int32, torch.int32),
                (torch.int64, torch.int32),
                (torch.int32, torch.int64),
            ],
            torch.int64: [
                (torch.int64, torch.int64),
                (torch.int32, torch.int64),
                (torch.int64, torch.int32),
            ],
        }

        supported_types = promotion_table.get(output_spec.dtype, [(output_spec.dtype, output_spec.dtype)])
        dtypes = random.choice(supported_types)

        return [
            ScalarSpec(dtype=dtypes[0]),
            ScalarSpec(dtype=dtypes[1])
        ]

    def codegen(self, output_name: str, input_names: List[str], output_spec: Spec) -> str:
        """Generate code for scalar multiplication operation."""
        if len(input_names) != 2:
            raise ValueError("ScalarMultiplyOperator requires exactly two inputs")

        return f"{output_name} = {input_names[0]} * {input_names[1]}"
