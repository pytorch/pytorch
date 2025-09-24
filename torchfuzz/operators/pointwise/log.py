"""Log operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class LogOperator(Operator):
    """Operator for logarithm function."""

    def __init__(self):
        super().__init__("log")

    def can_produce(self, tensor):
        """Log can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Log."""
        # The input to Log must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Log operation."""
        # Use torch.log for the logarithm operation
        return f"{output_name} = torch.log({input_names[0]})"
