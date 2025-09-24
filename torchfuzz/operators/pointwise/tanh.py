"""Tanh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class TanhOperator(Operator):
    """Operator for Tanh activation function."""

    def __init__(self):
        super().__init__("tanh")

    def can_produce(self, tensor):
        """Tanh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Tanh."""
        # The input to Tanh must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Tanh operation."""
        # Use torch.tanh for the Tanh activation
        return f"{output_name} = torch.tanh({input_names[0]})"
