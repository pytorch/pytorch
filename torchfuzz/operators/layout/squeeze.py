"""Squeeze operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class SqueezeOperator(Operator):
    """Operator for tensor squeeze operations."""

    def __init__(self):
        super().__init__("squeeze")

    def can_produce(self, tensor):
        """Squeeze can only produce tensors that are the result of removing dimensions of size 1."""
        # A tensor can be produced by squeeze if we can find a valid input tensor
        # that, when squeezed, results in the output tensor
        # This op can always produce a given shape by inserting a size-1 dim and then squeezing it.
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for squeeze operation."""
        output_shape = tensor.size

        # Strategy: Add exactly one dimension of size 1 to the output shape
        # This ensures squeeze removes exactly one dimension
        insert_pos = random.randint(0, len(output_shape))
        input_shape_list = list(output_shape)
        input_shape_list.insert(insert_pos, 1)
        input_shape = tuple(input_shape_list)

        # Calculate contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(input_shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        # Store which dimension to squeeze on the OUTPUT tensor to avoid shared-operator state bugs
        tensor._squeeze_dim = insert_pos

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for squeeze operation."""
        # Read the squeeze dim from the output tensor metadata (set during decompose)
        dim = getattr(output_tensor, "_squeeze_dim", None)
        if dim is not None:
            return f"{output_name} = torch.squeeze({input_names[0]}, {dim})"
        # Fallback: parameterless squeeze (removes all size-1 dims)
        return f"{output_name} = torch.squeeze({input_names[0]})"
