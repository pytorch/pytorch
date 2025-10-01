"""Unsqueeze operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class UnsqueezeOperator(Operator):
    """Operator for tensor unsqueeze operations."""

    def __init__(self):
        super().__init__("unsqueeze")

    def can_produce(self, tensor):
        """Unsqueeze can only produce tensors that have at least one dimension of size 1."""
        # Unsqueeze adds dimensions of size 1, so it can only produce tensors that have
        # at least one dimension of size 1 (that was added by the unsqueeze operation)
        # Exception: scalars can be produced from empty tensors
        if len(tensor.size) == 0:
            return True

        # Check if tensor has at least one dimension of size 1
        return any(dim == 1 for dim in tensor.size)

    def decompose(self, tensor):
        """Decompose tensor into input tensor for unsqueeze operation."""
        output_shape = tensor.size

        # Handle scalar output - create empty tensor input and unsqueeze at dim 0
        if len(output_shape) == 0:
            input_shape = ()
            unsqueeze_dim = 0
        else:
            # Strategy: Only create valid inputs for unsqueeze
            # Unsqueeze ALWAYS adds a dimension of size 1, so we can only "decompose"
            # tensors that have at least one dimension of size 1
            ones_positions = [i for i, dim in enumerate(output_shape) if dim == 1]

            if ones_positions:
                # Remove one of the size-1 dimensions to create input
                remove_pos = random.choice(ones_positions)
                input_shape = tuple(dim for i, dim in enumerate(output_shape) if i != remove_pos)
                unsqueeze_dim = remove_pos
            else:
                # This should not happen since can_produce should filter out tensors
                # without size-1 dimensions, but handle it gracefully
                # We'll just remove the first dimension and use it as unsqueeze position
                input_shape = output_shape[1:] if len(output_shape) > 1 else ()
                unsqueeze_dim = 0

        # Calculate contiguous stride for input
        if len(input_shape) == 0:
            stride = ()
        else:
            stride = []
            acc = 1
            for s in reversed(input_shape):
                stride.insert(0, acc)
                acc *= s
            stride = tuple(stride)

        t_in = Tensor(input_shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        t_in._unsqueeze_dim = unsqueeze_dim
        self._last_input_tensor = t_in

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for unsqueeze operation."""
        # The unsqueeze dimension should be stored in the input tensor metadata
        input_tensor = self._last_input_tensor if hasattr(self, '_last_input_tensor') else None

        if input_tensor and hasattr(input_tensor, '_unsqueeze_dim'):
            dim = input_tensor._unsqueeze_dim
        else:
            # Fallback: use dimension 0
            dim = 0

        # output_tensor parameter is needed for the interface but not used in this implementation
        return f"{output_name} = torch.unsqueeze({input_names[0]}, {dim})"
