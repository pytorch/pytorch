"""Mul operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MulOperator(Operator):
    """Operator for element-wise multiplication."""

    def __init__(self):
        super().__init__("mul")

    def can_produce(self, tensor):
        """Mul can always produce a tensor by multiplying two tensors of the same shape, dtype, etc."""
        return True

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for multiplication with type promotion."""
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
        }
        # If num_inputs > 2, promote left-to-right (e.g. (((a * b) * c) * d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype = tensor.dtype
        supported_types = promotion_table.get(dtype, [(dtype, dtype)])
        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype)
        else:
            dtypes = [dtype] * num_inputs

        return [
            Tensor(tensor.size, tensor.stride, dt, tensor.device, tensor.supported_ops)
            for dt in dtypes
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for multiplication operation."""
        # Multiply all input tensors
        expr = " * ".join(input_names)
        return f"{output_name} = {expr}"
