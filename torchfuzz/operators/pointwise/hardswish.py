"""Hardswish operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class HardswishOperator(Operator):
    """Operator for Hardswish activation function (torch.nn.functional.hardswish)."""

    def __init__(self):
        super().__init__("hardswish")

    def can_produce(self, tensor):
        """Hardswish can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Hardswish."""
        # The input to Hardswish must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Hardswish operation."""
        return f"{output_name} = torch.nn.functional.hardswish({input_names[0]})"
