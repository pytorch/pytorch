"""Upper triangular operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class TriuOperator(Operator):
    """Operator for extracting upper triangular part of matrices."""

    def __init__(self):
        super().__init__("triu")

    def can_produce(self, tensor):
        """Triu can produce matrices from matrices (preserves shape) with at least 2 dimensions."""
        # Triu needs at least 2 dimensions to work on matrices
        # and works on any numeric type
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input tensor for upper triangular operation."""
        # Triu preserves the shape of the input matrix
        # Input shape: (..., m, n) -> Output shape: (..., m, n)

        if len(tensor.size) < 2:
            # This shouldn't happen due to can_produce check, but handle it
            raise ValueError("Triu requires tensors with at least 2 dimensions")

        # Input has the same shape as output
        input_size = tensor.size

        # Store diagonal offset parameter for codegen
        self._diagonal = random.randint(-2, 2)  # Random diagonal offset

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
        """Generate code for upper triangular operation."""
        if len(input_names) != 1:
            raise ValueError("triu requires exactly 1 input")

        diagonal = getattr(self, '_diagonal', 0)

        # Generate different code based on diagonal offset
        if diagonal == 0:
            # Default case (main diagonal)
            return f"{output_name} = torch.triu({input_names[0]})"
        else:
            # With diagonal offset
            return f"{output_name} = torch.triu({input_names[0]}, diagonal={diagonal})"
