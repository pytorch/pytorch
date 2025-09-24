"""Rsqrt operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RsqrtOperator(Operator):
    """Operator for reciprocal square root function."""

    def __init__(self):
        super().__init__("rsqrt")

    def can_produce(self, tensor):
        """Rsqrt can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Rsqrt."""
        # The input to Rsqrt must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Rsqrt operation."""
        # Use torch.rsqrt for the reciprocal square root operation
        return f"{output_name} = torch.rsqrt({input_names[0]})"
