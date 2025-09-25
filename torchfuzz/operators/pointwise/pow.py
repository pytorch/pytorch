"""Pow operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class PowOperator(Operator):
    """Operator for element-wise power operation. DTensor-safe operation."""

    def __init__(self):
        super().__init__(supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """Pow can always produce a tensor by raising a tensor to a power."""
        return True

    def supports_variable_inputs(self):
        """Pow operator supports variable number of inputs."""
        return True

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for power operation with type promotion."""
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
        # For power operation, we typically have base^exponent
        # If num_inputs > 2, we chain left-to-right: ((a^b)^c)^d
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
        """Generate code for power operation."""
        # Use torch.pow for power operation
        if len(input_names) == 1:
            # Square the input if only one input
            return f"{output_name} = torch.pow({input_names[0]}, 2.0)"
        else:
            # Chain powers left-to-right
            expr = input_names[0]
            for name in input_names[1:]:
                expr = f"torch.pow({expr}, {name})"
            return f"{output_name} = {expr}"
