"""Sqrt operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor
from torchfuzz.type_promotion import is_integer_dtype


class SqrtOperator(Operator):
    """Operator for square root operation."""

    def __init__(self):
        super().__init__("sqrt")

    def can_produce(self, tensor):
        """Sqrt can only produce float tensors (promotes integers to float)."""
        # torch.sqrt promotes integer tensors to float, so it cannot produce integer tensors
        if is_integer_dtype(tensor.dtype):
            return False
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Sqrt with proper type promotion."""
        # torch.sqrt can take any numeric input, but promotes integers to float
        # Choose an input dtype that makes sense
        if tensor.dtype == "float32":
            input_dtype = "int64"  # This will be promoted to float32
        else:
            input_dtype = tensor.dtype  # Keep same dtype for other float types

        return [
            Tensor(tensor.size, tensor.stride, input_dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Sqrt operation."""
        # Use torch.sqrt for the square root operation
        return f"{output_name} = torch.sqrt({input_names[0]})"
