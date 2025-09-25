"""ReLU operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ReluOperator(Operator):
    """Operator for ReLU activation function."""

    def __init__(self):
        super().__init__(supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """ReLU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for ReLU."""
        # The input to ReLU must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for ReLU operation."""
        # Use torch.nn.functional.relu for the ReLU activation
        return f"{output_name} = torch.nn.functional.relu({input_names[0]})"
