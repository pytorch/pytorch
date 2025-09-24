"""Atan operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AtanOperator(Operator):
    """Operator for arctangent function."""

    def __init__(self):
        super().__init__("atan")

    def can_produce(self, tensor):
        """Atan can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Atan."""
        # The input to Atan must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Atan operation."""
        # Use torch.atan for the arctangent operation
        return f"{output_name} = torch.atan({input_names[0]})"
