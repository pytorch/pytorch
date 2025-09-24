"""Erf operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ErfOperator(Operator):
    """Operator for error function."""

    def __init__(self):
        super().__init__("erf")

    def can_produce(self, tensor):
        """Erf can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Erf."""
        # The input to Erf must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Erf operation."""
        # Use torch.erf for the error function operation
        return f"{output_name} = torch.erf({input_names[0]})"
