"""Abs operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AbsOperator(Operator):
    """Operator for absolute value function."""

    def __init__(self):
        super().__init__("abs")

    def can_produce(self, tensor):
        """Abs can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Abs."""
        # The input to Abs must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Abs operation."""
        # Use torch.abs for the absolute value operation
        return f"{output_name} = torch.abs({input_names[0]})"
