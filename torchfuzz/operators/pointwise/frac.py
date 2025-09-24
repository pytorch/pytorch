"""Frac operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class FracOperator(Operator):
    """Operator for frac function."""

    def __init__(self):
        super().__init__("frac")

    def can_produce(self, tensor):
        """Frac can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Frac."""
        # The input to Frac must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Frac operation."""
        # Use torch.frac for the frac operation
        return f"{output_name} = torch.frac({input_names[0]})"
