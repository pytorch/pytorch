"""Atan2 operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class Atan2Operator(Operator):
    """Operator for atan2(y, x) function."""

    def __init__(self):
        super().__init__("atan2")

    def can_produce(self, tensor):
        """Atan2 can be applied to any tensor (binary elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensors for Atan2."""
        # Atan2 takes two tensors of the same shape, stride, dtype, and device as the output
        # First tensor is y, second tensor is x
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops),
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Atan2 operation."""
        # Use torch.atan2 for the atan2(y, x) operation
        if len(input_names) != 2:
            raise ValueError("Atan2 requires exactly 2 input tensors")
        return f"{output_name} = torch.atan2({input_names[0]}, {input_names[1]})"
