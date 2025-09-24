"""Permute operator implementation."""

import random
from ..base.operator import Operator
from torchfuzz.tensor import Tensor


class PermuteOperator(Operator):
    """Operator for tensor permutation operations."""

    def __init__(self):
        super().__init__("permute", supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """Permute can produce tensors with at least 1 dimension."""
        # Permute doesn't make sense for scalar tensors since they have no dimensions
        return len(output_tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensor for permute operation."""
        ndims = len(tensor.size)

        # Should never receive scalar tensors since _can_produce_impl returns False for them
        assert ndims >= 1, "PermuteOperator should not receive scalar tensors"

        # Generate a random permutation of dimensions
        dims = list(range(ndims))
        random.shuffle(dims)

        # Create input tensor - we need to invert the permutation
        # If output = input.permute(dims), then input[dims[i]] = output[i]
        # So input[j] = output[dims.index(j)]
        input_size = [0] * ndims
        input_stride = [0] * ndims

        for i in range(ndims):
            input_size[dims[i]] = tensor.size[i]
            input_stride[dims[i]] = tensor.stride[i]

        input_size = tuple(input_size)
        input_stride = tuple(input_stride)

        t_in = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Store the permutation dimensions on the output tensor
        tensor._permute_dims = tuple(dims)

        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for permute operation."""
        dims = getattr(output_tensor, "_permute_dims", tuple(range(len(output_tensor.size))))

        dims_str = ", ".join(str(d) for d in dims)
        return f"{output_name} = {input_names[0]}.permute({dims_str})"
