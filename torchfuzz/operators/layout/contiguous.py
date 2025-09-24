"""Contiguous operator implementation."""

import random
from ..base.operator import Operator
from torchfuzz.tensor import Tensor


class ContiguousOperator(Operator):
    """Operator for tensor contiguous operations."""

    def __init__(self):
        super().__init__(supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """Contiguous can produce any tensor (as it makes tensor contiguous)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for contiguous operation."""
        # Create a non-contiguous input tensor with the same shape
        # but different strides that would require contiguous() to fix

        if len(tensor.size) == 0:
            # Scalar tensor - already contiguous, just return it
            t_in = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
            return [t_in]

        # Generate non-contiguous strides by making them non-standard
        # We'll create strides that are valid but not in standard row-major order
        input_stride = list(tensor.stride)

        # Randomize strides while keeping them valid for the given shape
        # Method: multiply each stride by a random factor > 1
        for i in range(len(input_stride)):
            if input_stride[i] > 0:  # Don't modify zero strides (size-1 dimensions)
                factor = random.randint(2, 5)  # Random multiplier to make non-contiguous
                input_stride[i] *= factor

        # Alternative method for tensors with more than 1 dimension:
        # Swap some strides to make it non-contiguous
        if len(input_stride) >= 2 and random.choice([True, False]):
            # Randomly swap two adjacent strides
            idx = random.randint(0, len(input_stride) - 2)
            # Only swap if both are non-zero
            if input_stride[idx] > 0 and input_stride[idx + 1] > 0:
                input_stride[idx], input_stride[idx + 1] = input_stride[idx + 1], input_stride[idx]

        input_stride = tuple(input_stride)

        t_in = Tensor(tensor.size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for contiguous operation."""
        return f"{output_name} = {input_names[0]}.contiguous()"
