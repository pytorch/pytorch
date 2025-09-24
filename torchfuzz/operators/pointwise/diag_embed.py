"""Diagonal embed operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class DiagEmbedOperator(Operator):
    """Operator for embedding vectors as diagonal matrices."""

    def __init__(self):
        super().__init__("diag_embed")

    def can_produce(self, tensor):
        """DiagEmbed can produce square matrices from vectors with at least 2 dimensions."""
        # DiagEmbed needs at least 2 dimensions for output (square matrix)
        # and works on any numeric type
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input tensor for diagonal embedding."""
        # DiagEmbed creates diagonal matrices from vectors
        # Input shape: (..., n) -> Output shape: (..., n, n)
        # So input has one less dimension than output

        if len(tensor.size) < 2:
            # This shouldn't happen due to can_produce check, but handle it
            raise ValueError("DiagEmbed output must have at least 2 dimensions")

        # Remove one of the trailing dimensions to get input shape
        # The last two dimensions should be equal for square matrix
        if tensor.size[-1] != tensor.size[-2]:
            # Not a square matrix, can't be produced by diag_embed
            # Return a reasonable input anyway
            input_size = tensor.size[:-1]
        else:
            # Square matrix, remove last dimension
            input_size = tensor.size[:-1]

        # Store parameters for codegen
        self._offset = random.randint(-2, 2)  # Random diagonal offset
        self._dim1 = -2  # Default dimension for the diagonal
        self._dim2 = -1  # Default dimension for the diagonal

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
        """Generate code for diagonal embedding operation."""
        if len(input_names) != 1:
            raise ValueError("diag_embed requires exactly 1 input")

        offset = getattr(self, '_offset', 0)
        dim1 = getattr(self, '_dim1', -2)
        dim2 = getattr(self, '_dim2', -1)

        # Generate different code based on parameters
        if offset == 0 and dim1 == -2 and dim2 == -1:
            # Default case
            return f"{output_name} = torch.diag_embed({input_names[0]})"
        else:
            # With parameters
            return f"{output_name} = torch.diag_embed({input_names[0]}, offset={offset}, dim1={dim1}, dim2={dim2})"
