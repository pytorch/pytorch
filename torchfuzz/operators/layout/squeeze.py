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
        # tensor parameter is needed for the interface but not used in this implementation
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
        # Store which dimension to squeeze
        t_in._squeeze_dim = insert_pos
        self._last_input_tensor = t_in

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for squeeze operation."""
        # Get the dimension to squeeze from the stored metadata
        input_tensor = self._last_input_tensor if hasattr(self, '_last_input_tensor') else None
        
        if input_tensor and hasattr(input_tensor, '_squeeze_dim'):
            dim = input_tensor._squeeze_dim
            # Use the specific dimension we calculated during decomposition
            return f"{output_name} = torch.squeeze({input_names[0]}, {dim})"
        else:
            # Fallback: use parameterless squeeze (removes all size-1 dims)
            return f"{output_name} = torch.squeeze({input_names[0]})"
