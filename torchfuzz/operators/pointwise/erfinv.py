"""Erfinv operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ErfinvOperator(Operator):
    """Operator for inverse error function."""

    def __init__(self):
        super().__init__("erfinv")

    def can_produce(self, tensor):
        """Erfinv can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Erfinv."""
        # The input to Erfinv must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Erfinv operation."""
        # Use torch.erfinv for the inverse error function operation
        return f"{output_name} = torch.erfinv({input_names[0]})"
