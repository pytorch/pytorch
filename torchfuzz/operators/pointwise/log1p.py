"""Log1p operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class Log1pOperator(Operator):
    """Operator for log(1 + x) function."""

    def __init__(self):
        super().__init__("log1p")

    def can_produce(self, tensor):
        """Log1p can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Log1p."""
        # The input to Log1p must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Log1p operation."""
        # Use torch.log1p for the log(1 + x) operation
        return f"{output_name} = torch.log1p({input_names[0]})"
