"""Sqrt operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SqrtOperator(Operator):
    """Operator for square root operation."""

    def __init__(self):
        super().__init__(supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """Sqrt can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Sqrt."""
        # The input to Sqrt must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Sqrt operation."""
        # Use torch.sqrt for the square root operation
        return f"{output_name} = torch.sqrt({input_names[0]})"
