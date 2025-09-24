"""Sigmoid operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SigmoidOperator(Operator):
    """Operator for Sigmoid activation function."""

    def __init__(self):
        super().__init__("sigmoid")

    def can_produce(self, tensor):
        """Sigmoid can be applied to floating point tensors (elementwise op)."""
        # Sigmoid only supports floating point and complex dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Sigmoid."""
        # The input to Sigmoid must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Sigmoid operation."""
        # Use torch.sigmoid for the Sigmoid activation
        return f"{output_name} = torch.sigmoid({input_names[0]})"
