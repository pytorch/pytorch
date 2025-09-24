"""Local response normalization operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class LocalResponseNormOperator(Operator):
    """Operator for local response normalization (torch.nn.functional.local_response_norm)."""

    def __init__(self):
        super().__init__("local_response_norm")

    def can_produce(self, tensor):
        """LocalResponseNorm can produce tensors that are at least 3D (N, C, ...) with floating point types."""
        # Local response norm only supports floating point dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Local response norm requires at least 3D tensors (N, C, ...)
        # Also need sufficient channels for the window
        return len(tensor.size) >= 3 and tensor.size[1] >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensor for local response norm operation."""
        # The input to local response norm must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate size parameter (window size for normalization)
        # Common values are 3, 5, 7, 9 etc.
        size = random.choice([3, 5, 7, 9])

        # Store parameters for codegen
        self._size = size

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for local response norm operation."""
        size = getattr(self, '_size', 5)  # Default size
        return f"{output_name} = torch.nn.functional.local_response_norm({input_names[0]}, size={size})"
