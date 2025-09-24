"""Reciprocal operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ReciprocalOperator(Operator):
    """Operator for reciprocal function."""

    def __init__(self):
        super().__init__("reciprocal")

    def can_produce(self, tensor):
        """Reciprocal can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Reciprocal."""
        # The input to Reciprocal must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Reciprocal operation."""
        # Use torch.reciprocal for the reciprocal operation
        return f"{output_name} = torch.reciprocal({input_names[0]})"
