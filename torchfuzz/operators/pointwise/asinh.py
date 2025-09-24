"""Asinh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AsinhOperator(Operator):
    """Operator for inverse hyperbolic sine function."""

    def __init__(self):
        super().__init__("asinh")

    def can_produce(self, tensor):
        """Asinh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Asinh."""
        # The input to Asinh must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Asinh operation."""
        # Use torch.asinh for the inverse hyperbolic sine operation
        return f"{output_name} = torch.asinh({input_names[0]})"
