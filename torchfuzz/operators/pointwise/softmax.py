"""Softmax operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class SoftmaxOperator(Operator):
    """Operator for softmax activation (torch.nn.functional.softmax)."""

    def __init__(self):
        super().__init__("softmax")

    def can_produce(self, tensor):
        """Softmax can be applied to tensors with at least 1 dimension and floating point types."""
        # Softmax only supports floating point dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Softmax needs at least 1 dimension to work on
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensor for softmax."""
        # The input to softmax must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Choose a random dimension to apply softmax over
        # For softmax, we typically apply over the last dimension by default, but can vary
        ndim = len(tensor.size)
        dim = random.randint(-ndim, ndim - 1)  # Allow negative indexing

        # Store the dimension for codegen
        self._dim = dim

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for softmax operation."""
        dim = getattr(self, '_dim', -1)  # Default to last dimension
        return f"{output_name} = torch.nn.functional.softmax({input_names[0]}, dim={dim})"
