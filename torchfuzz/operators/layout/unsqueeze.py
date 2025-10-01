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
        # Unsqueeze adds dimensions of size 1, so it can only produce tensors that:
        # 1. Have at least one dimension of size 1 (which could have been added by unsqueeze)
        # 2. Are not scalars (unsqueeze always increases rank)
        
        if len(tensor.size) == 0:
            return False
        
        # Only produce tensors that have at least one dimension of size 1
        return any(dim == 1 for dim in tensor.size)

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
            # No size-1 dimensions in output. This should not happen since can_produce
            # should filter out tensors without size-1 dimensions.
            # But if it does happen, we'll handle it by treating this as an error case.
            raise ValueError(f"UnsqueezeOperator cannot produce tensor with shape {output_shape} "
                           f"because it has no dimensions of size 1. This indicates a bug in can_produce().")

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
