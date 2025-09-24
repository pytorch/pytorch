"""Log10 operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class Log10Operator(Operator):
    """Operator for base-10 logarithm function."""

    def __init__(self):
        super().__init__("log10")

    def can_produce(self, tensor):
        """Log10 can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Log10."""
        # The input to Log10 must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Log10 operation."""
        # Use torch.log10 for the base-10 logarithm operation
        return f"{output_name} = torch.log10({input_names[0]})"
