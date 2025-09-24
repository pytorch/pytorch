"""Acos operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class AcosOperator(Operator):
    """Operator for arccosine function."""

    def __init__(self):
        super().__init__("acos")

    def can_produce(self, tensor):
        """Acos can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Acos."""
        # The input to Acos must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Acos operation."""
        # Use torch.acos for the arccosine operation
        return f"{output_name} = torch.acos({input_names[0]})"
