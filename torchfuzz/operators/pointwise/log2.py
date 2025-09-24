"""Log2 operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class Log2Operator(Operator):
    """Operator for base-2 logarithm function."""

    def __init__(self):
        super().__init__("log2")

    def can_produce(self, tensor):
        """Log2 can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Log2."""
        # The input to Log2 must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Log2 operation."""
        # Use torch.log2 for the base-2 logarithm operation
        return f"{output_name} = torch.log2({input_names[0]})"
