"""Neg operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class NegOperator(Operator):
    """Operator for negation function."""

    def __init__(self):
        super().__init__("neg")

    def can_produce(self, tensor):
        """Neg can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Neg."""
        # The input to Neg must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Neg operation."""
        # Use torch.neg for the negation operation
        return f"{output_name} = torch.neg({input_names[0]})"
