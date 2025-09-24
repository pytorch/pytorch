"""Exp operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ExpOperator(Operator):
    """Operator for exponential function."""

    def __init__(self):
        super().__init__("exp")

    def can_produce(self, tensor):
        """Exp can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Exp."""
        # The input to Exp must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Exp operation."""
        # Use torch.exp for the exponential operation
        return f"{output_name} = torch.exp({input_names[0]})"
