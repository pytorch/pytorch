"""Dropout operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class DropoutOperator(Operator):
    """Operator for dropout (torch.nn.functional.dropout)."""

    def __init__(self):
        super().__init__("dropout")

    def can_produce(self, tensor):
        """Dropout can be applied to any tensor (elementwise op) with floating point dtypes."""
        allowed_dtypes = {"float32", "float64", "bfloat16", "float16"}
        return str(tensor.dtype).lower() in allowed_dtypes

    def decompose(self, tensor):
        """Decompose tensor into input tensor for dropout."""
        # The input to dropout must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for dropout operation."""
        # Choose a random dropout probability
        p = random.choice([0.1, 0.2, 0.3, 0.5])
        training = random.choice([True, False])
        return f"{output_name} = torch.nn.functional.dropout({input_names[0]}, p={p}, training={training})"
