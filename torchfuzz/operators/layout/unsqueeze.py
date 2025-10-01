"""Unsqueeze operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class UnsqueezeOperator(Operator):
    """Operator for tensor unsqueeze operations."""

    def __init__(self):
        super().__init__("unsqueeze")

    def can_produce(self, tensor):
        """Unsqueeze can produce tensors by adding dimensions of size 1."""
        # Unsqueeze adds dimensions of size 1, so it can produce:
        # 1. Any tensor that has at least one dimension of size 1 (remove that dimension for input)
        # 2. Any tensor by adding a new dimension of size 1 somewhere
        
        # Special case: scalars cannot be produced by unsqueeze (unsqueeze always increases rank)
        if len(tensor.size) == 0:
            return False
        
        # All non-scalar tensors can potentially be produced by unsqueeze
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for unsqueeze operation."""
        output_shape = tensor.size

        # Since can_produce returns False for scalars, we should never get here with a scalar
        assert len(output_shape) > 0, "UnsqueezeOperator should not receive scalar tensors"
          
        # Strategy: For unsqueeze to produce output_shape, we need an input that when
        # unsqueezed at some dimension results in output_shape
        ones_positions = [i for i, dim in enumerate(output_shape) if dim == 1]

        if ones_positions:
            # Remove one of the size-1 dimensions to create input
            # This represents the dimension that was added by unsqueeze
            remove_pos = random.choice(ones_positions)
            input_shape = tuple(dim for i, dim in enumerate(output_shape) if i != remove_pos)
            unsqueeze_dim = remove_pos
        else:
            # No size-1 dimensions in output, so we need to create an input that when
            # unsqueezed at some position will produce the output shape.
            # Since there are no size-1 dims, we need to remove one dimension from output.
            # But this is problematic because unsqueeze only adds dimensions.
            # For this case, we'll create a smaller input tensor by removing the last dimension
            if len(output_shape) == 1:
                # If output is 1D with no size-1 dims, create a scalar input
                input_shape = ()
                unsqueeze_dim = 0
            else:
                # Remove the last dimension to create input
                input_shape = output_shape[:-1]
                unsqueeze_dim = len(input_shape)  # Insert at the end

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
