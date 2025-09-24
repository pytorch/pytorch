"""Tan operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class TanOperator(Operator):
    """Operator for tangent function."""

    def __init__(self):
        super().__init__("tan")

    def can_produce(self, tensor):
        """Tan can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Tan."""
        # The input to Tan must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Tan operation."""
        # Use torch.tan for the tangent operation
        return f"{output_name} = torch.tan({input_names[0]})"
