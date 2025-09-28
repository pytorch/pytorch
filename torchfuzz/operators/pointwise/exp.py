"""Exp operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor
from torchfuzz.type_promotion import get_promoted_dtype, can_produce_integer_tensor, is_integer_dtype


class ExpOperator(Operator):
    """Operator for exponential function."""

    def __init__(self):
        super().__init__("exp")

    def can_produce(self, tensor):
        """Exp can only produce float tensors (promotes integers to float)."""
        # torch.exp promotes integer tensors to float, so it cannot produce integer tensors
        if is_integer_dtype(tensor.dtype):
            return False
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Exp with proper type promotion."""
        # torch.exp can take any numeric input, but promotes integers to float
        # We need to find an input dtype that would produce the desired output dtype

        # If the output is float, the input could be integer or float
        # We'll allow both integer and float inputs, knowing that integers will be promoted
        possible_input_dtypes = ["int64", "float32", "float16", "bfloat16"]

        # Choose an input dtype that makes sense
        # If output is float32, prefer int64 input to demonstrate promotion
        # Otherwise, match the output dtype
        if tensor.dtype == "float32":
            input_dtype = "int64"  # This will be promoted to float32
        else:
            input_dtype = tensor.dtype  # Keep same dtype for other float types

        return [
            Tensor(tensor.size, tensor.stride, input_dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Exp operation."""
        # Use torch.exp for the exponential operation
        return f"{output_name} = torch.exp({input_names[0]})"
