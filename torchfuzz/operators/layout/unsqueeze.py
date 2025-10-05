"""Unsqueeze operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class UnsqueezeOperator(Operator):
    """Operator for tensor unsqueeze operations."""

    def __init__(self):
        super().__init__("unsqueeze")

    def can_produce(self, tensor):
        """Unsqueeze produces tensors that have at least one dimension of size 1 and are non-scalars."""
        # Unsqueeze increases rank by inserting a size-1 dimension.
        # Therefore, any non-scalar tensor that contains at least one size-1 dimension
        # could have been produced by unsqueezing a lower-rank tensor.
        return len(tensor.size) > 0 and any(dim == 1 for dim in tensor.size)

    def decompose(self, tensor):
        """Decompose tensor into input tensor for unsqueeze operation."""
        output_shape = tensor.size
        assert len(output_shape) > 0

        # Choose a size-1 dimension in the output that could have been inserted by unsqueeze
        ones_positions = [i for i, dim in enumerate(output_shape) if dim == 1]
        if not ones_positions:
            # Should not happen due to can_produce, but guard anyway
            raise ValueError(f"UnsqueezeOperator cannot produce shape {output_shape} without a size-1 dim")
        insert_pos = random.choice(ones_positions)

        # Remove that dimension to get the input shape
        input_shape = tuple(dim for i, dim in enumerate(output_shape) if i != insert_pos)

        # Calculate contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape if len(input_shape) > 0 else ()):  # scalar case handled
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(input_shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        # Store metadata on the OUTPUT tensor to avoid operator-instance state bugs
        tensor._unsqueeze_dim = insert_pos

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for unsqueeze operation."""
        dim = getattr(output_tensor, "_unsqueeze_dim", None)
        if dim is None:
            dim = 0
        return f"{output_name} = torch.unsqueeze({input_names[0]}, {dim})"
