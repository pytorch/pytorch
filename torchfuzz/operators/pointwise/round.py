"""Round operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RoundOperator(Operator):
    """Operator for round function."""

    def __init__(self):
        super().__init__("round")

    def can_produce(self, tensor):
        """Round can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Round."""
        # The input to Round must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Round operation."""
        # Use torch.round for the round operation
        return f"{output_name} = torch.round({input_names[0]})"
