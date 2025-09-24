"""Diagonal operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class DiagOperator(Operator):
    """Operator for extracting diagonal from matrix or creating diagonal matrix from vector."""

    def __init__(self):
        super().__init__("diag")

    def can_produce(self, tensor):
        """Diag can produce vectors from matrices or matrices from vectors."""
        # Diag works on any numeric type
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for diagonal operation."""
        # Two modes:
        # 1. Extract diagonal: input is (..., n, n) -> output is (..., n)
        # 2. Create diagonal: input is (..., n) -> output is (..., n, n)

        # Decide which mode based on output shape
        if len(tensor.size) >= 2 and tensor.size[-1] == tensor.size[-2]:
            # Output is square matrix, so input should be a vector (create diagonal mode)
            # Remove the last dimension to get vector input
            input_size = tensor.size[:-1]
            self._mode = "create"
        else:
            # Output is vector, so input should be a square matrix (extract diagonal mode)
            # Add a dimension to make it square
            n = tensor.size[-1] if tensor.size else random.randint(3, 6)
            input_size = tensor.size + (n,)
            self._mode = "extract"

        # Calculate stride for contiguous tensor
        def calc_stride(size):
            if not size:
                return ()
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for diagonal operation."""
        if len(input_names) != 1:
            raise ValueError("diag requires exactly 1 input")

        return f"{output_name} = torch.diag({input_names[0]})"
