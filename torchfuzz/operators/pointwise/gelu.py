"""GELU operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class GeluOperator(Operator):
    """Operator for GELU activation function."""

    def __init__(self):
        super().__init__("gelu")

    def can_produce(self, tensor):
        """GELU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for GELU."""
        # The input to GELU must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for GELU operation."""
        # Use torch.nn.functional.gelu for the GELU activation
        return f"{output_name} = torch.nn.functional.gelu({input_names[0]})"
