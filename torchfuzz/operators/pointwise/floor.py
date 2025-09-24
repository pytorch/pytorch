"""Floor operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class FloorOperator(Operator):
    """Operator for floor function."""

    def __init__(self):
        super().__init__("floor")

    def can_produce(self, tensor):
        """Floor can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Floor."""
        # The input to Floor must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Floor operation."""
        # Use torch.floor for the floor operation
        return f"{output_name} = torch.floor({input_names[0]})"
