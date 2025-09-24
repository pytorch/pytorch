"""Cosh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class CoshOperator(Operator):
    """Operator for hyperbolic cosine function."""

    def __init__(self):
        super().__init__("cosh")

    def can_produce(self, tensor):
        """Cosh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Cosh."""
        # The input to Cosh must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Cosh operation."""
        # Use torch.cosh for the hyperbolic cosine operation
        return f"{output_name} = torch.cosh({input_names[0]})"
