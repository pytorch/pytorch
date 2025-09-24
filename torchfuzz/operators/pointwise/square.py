"""Square operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SquareOperator(Operator):
    """Operator for square function."""

    def __init__(self):
        super().__init__("square")

    def can_produce(self, tensor):
        """Square can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Square."""
        # The input to Square must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Square operation."""
        # Use torch.square for the square operation
        return f"{output_name} = torch.square({input_names[0]})"
