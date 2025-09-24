"""Sin operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SinOperator(Operator):
    """Operator for sine function."""

    def __init__(self):
        super().__init__("sin")

    def can_produce(self, tensor):
        """Sin can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Sin."""
        # The input to Sin must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Sin operation."""
        # Use torch.sin for the sine operation
        return f"{output_name} = torch.sin({input_names[0]})"
