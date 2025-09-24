"""Cos operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class CosOperator(Operator):
    """Operator for cosine function."""

    def __init__(self):
        super().__init__("cos")

    def can_produce(self, tensor):
        """Cos can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Cos."""
        # The input to Cos must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Cos operation."""
        # Use torch.cos for the cosine operation
        return f"{output_name} = torch.cos({input_names[0]})"
