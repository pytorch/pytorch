"""Multiply operator implementation."""

import random
from typing import List
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import TensorSpec, Spec


class MulOperator(Operator):
    """Operator for element-wise multiplication."""

    def __init__(self):
        super().__init__("torch.ops.aten.mul")

    def can_produce(self, output_spec: Spec) -> bool:
        """Mul can produce tensors but not scalars."""
        return isinstance(output_spec, TensorSpec)

    def supports_variable_inputs(self) -> bool:
        """Mul operator supports variable number of inputs."""
        return True

    def decompose(self, output_spec: Spec, num_inputs: int = 2) -> List[Spec]:
        """Decompose tensor into input tensors for multiplication with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("MulOperator can only produce TensorSpec outputs")

        # Type promotion table for realistic LLM/diffusion model types
        # Each output dtype maps to possible input dtype pairs (in order of preference)
        promotion_table = {
            "float32": [
                ("float32", "float32"),
                ("bfloat16", "float32"),
                ("float32", "bfloat16"),
                ("float16", "float32"),
                ("float32", "float16"),
            ],
            "bfloat16": [
                ("bfloat16", "bfloat16"),
                ("float32", "bfloat16"),
                ("bfloat16", "float32"),
            ],
            "float16": [
                ("float16", "float16"),
                ("float32", "float16"),
                ("float16", "float32"),
            ],
        }

        # If num_inputs > 2, promote left-to-right (e.g. (((a * b) * c) * d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype_str = str(output_spec.dtype).split(".")[-1]  # Get dtype name
        supported_types = promotion_table.get(dtype_str, [(dtype_str, dtype_str)])

        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype_str)
        else:
            dtypes = [dtype_str] * num_inputs

        # Convert dtype strings back to torch dtypes
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype_map.get(dt, output_spec.dtype)
            )
            for dt in dtypes
        ]

    def codegen(self, output_name: str, input_names: List[str], output_spec: Spec) -> str:
        """Generate code for multiplication operation."""
        if len(input_names) == 2:
            return f"{output_name} = torch.ops.aten.mul({input_names[0]}, {input_names[1]})"
        else:
            # Multiply all input tensors
            expr = " * ".join(input_names)
            return f"{output_name} = {expr}"
