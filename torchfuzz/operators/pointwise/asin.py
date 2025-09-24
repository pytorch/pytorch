"""Asin operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AsinOperator(Operator):
    """Operator for arcsine function."""

    def __init__(self):
        super().__init__("asin")

    def can_produce(self, tensor):
        """Asin can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Asin."""
        # The input to Asin must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Asin operation."""
        # Use torch.asin for the arcsine operation
        return f"{output_name} = torch.asin({input_names[0]})"
