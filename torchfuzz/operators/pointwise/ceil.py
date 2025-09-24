"""Ceil operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class CeilOperator(Operator):
    """Operator for ceil function."""

    def __init__(self):
        super().__init__("ceil")

    def can_produce(self, tensor):
        """Ceil can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Ceil."""
        # The input to Ceil must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Ceil operation."""
        # Use torch.ceil for the ceil operation
        return f"{output_name} = torch.ceil({input_names[0]})"
