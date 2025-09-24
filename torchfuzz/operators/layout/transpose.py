"""Transpose operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class TransposeOperator(Operator):
    """Operator for tensor transpose operations."""

    def __init__(self):
        super().__init__("transpose")

    def can_produce(self, tensor):
        """Transpose can always produce any tensor by transposing another tensor."""
        # Can only transpose tensors with at least 2 dimensions
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input tensor for transpose operation."""
        if len(tensor.size) < 2:
            raise ValueError("Cannot transpose tensor with less than 2 dimensions")

        # Pick two dimensions to transpose
        ndims = len(tensor.size)
        dim0 = random.randint(0, ndims - 1)
        dim1 = random.randint(0, ndims - 1)
        # Make sure we pick different dimensions
        while dim0 == dim1 and ndims > 1:
            dim1 = random.randint(0, ndims - 1)

        # Create input tensor with dimensions swapped
        input_size = list(tensor.size)
        input_size[dim0], input_size[dim1] = input_size[dim1], input_size[dim0]
        input_size = tuple(input_size)

        # Create corresponding strides (also swapped)
        input_stride = list(tensor.stride)
        input_stride[dim0], input_stride[dim1] = input_stride[dim1], input_stride[dim0]
        input_stride = tuple(input_stride)

        t_in = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Store the dimensions to transpose on the output tensor
        tensor._transpose_dim0 = dim0
        tensor._transpose_dim1 = dim1

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for transpose operation."""
        dim0 = getattr(output_tensor, "_transpose_dim0", 0)
        dim1 = getattr(output_tensor, "_transpose_dim1", 1)
        return f"{output_name} = {input_names[0]}.transpose({dim0}, {dim1})"
