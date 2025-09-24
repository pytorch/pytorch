"""Atanh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AtanhOperator(Operator):
    """Operator for inverse hyperbolic tangent function."""

    def __init__(self):
        super().__init__("atanh")

    def can_produce(self, tensor):
        """Atanh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Atanh."""
        # The input to Atanh must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Atanh operation."""
        # Use torch.atanh for the inverse hyperbolic tangent operation
        return f"{output_name} = torch.atanh({input_names[0]})"
