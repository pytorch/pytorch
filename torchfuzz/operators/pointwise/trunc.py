"""Trunc operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class TruncOperator(Operator):
    """Operator for trunc function."""

    def __init__(self):
        super().__init__("trunc")

    def can_produce(self, tensor):
        """Trunc can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Trunc."""
        # The input to Trunc must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Trunc operation."""
        # Use torch.trunc for the trunc operation
        return f"{output_name} = torch.trunc({input_names[0]})"
