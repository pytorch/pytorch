"""Acosh operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AcoshOperator(Operator):
    """Operator for inverse hyperbolic cosine function."""

    def __init__(self):
        super().__init__("acosh")

    def can_produce(self, tensor):
        """Acosh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Acosh."""
        # The input to Acosh must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Acosh operation."""
        # Use torch.acosh for the inverse hyperbolic cosine operation
        return f"{output_name} = torch.acosh({input_names[0]})"
