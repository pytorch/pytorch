"""Sinh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SinhOperator(Operator):
    """Operator for hyperbolic sine function."""

    def __init__(self):
        super().__init__("sinh")

    def can_produce(self, tensor):
        """Sinh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Sinh."""
        # The input to Sinh must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Sinh operation."""
        # Use torch.sinh for the hyperbolic sine operation
        return f"{output_name} = torch.sinh({input_names[0]})"
