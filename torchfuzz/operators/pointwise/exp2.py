"""Exp2 operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class Exp2Operator(Operator):
    """Operator for base-2 exponential function."""

    def __init__(self):
        super().__init__("exp2")

    def can_produce(self, tensor):
        """Exp2 can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Exp2."""
        # The input to Exp2 must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Exp2 operation."""
        # Use torch.exp2 for the base-2 exponential operation
        return f"{output_name} = torch.exp2({input_names[0]})"
