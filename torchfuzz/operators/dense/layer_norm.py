"""Layer normalization operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class LayerNormOperator(Operator):
    """Operator for layer normalization (torch.nn.functional.layer_norm)."""

    def __init__(self):
        super().__init__("layer_norm")

    def can_produce(self, tensor):
        """LayerNorm can produce tensors of various dimensions and floating point types."""
        # LayerNorm only supports floating point and complex dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input tensors for layer norm operation."""

        # tensor shape is the output shape
        input_size = tensor.size

        # normalized_shape is typically the last few dimensions
        # Common patterns: last dim, last 2 dims, etc.
        # For stability, let's prefer smaller normalized shapes that are more common
        ndim = len(input_size)
        if ndim >= 3:
            # For 3D+ tensors, typically normalize over last 1 or 2 dimensions
            norm_dims = random.choice([1, 2])
        else:
            # For 2D tensors, normalize over last dimension
            norm_dims = 1
        # Ensure normalized_shape matches the last norm_dims and is valid for input_size
        normalized_shape = input_size[-norm_dims:]

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        # Type promotion for realistic LLM types
        dtype = tensor.dtype

        # Create input tensor (same shape as output)
        input_tensor = Tensor(input_size, input_stride, dtype, tensor.device, tensor.supported_ops)

        # normalized_shape as a tensor (representing the shape tuple)
        shape_tensor = Tensor((len(normalized_shape),), (1,), "int64", tensor.device, tensor.supported_ops)

        result = [input_tensor, shape_tensor]

        # Store normalized_shape for codegen
        self._normalized_shape = normalized_shape
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for layer norm operation."""
        if len(input_names) not in [2]:
            raise ValueError("Layer norm requires 2 inputs")

        # Get the normalized shape
        normalized_shape = getattr(self, '_normalized_shape', output_tensor.size[-1:])
        # Validate that normalized_shape matches the last dimensions of the input tensor
        # If not, fallback to a valid normalized_shape
        input_shape = output_tensor.size
        if len(normalized_shape) > len(input_shape) or normalized_shape != input_shape[-len(normalized_shape):]:
            # Fallback: use the last dimension as normalized_shape
            normalized_shape = (input_shape[-1],)
        shape_str = str(normalized_shape)

        return f"{output_name} = torch.nn.functional.layer_norm({input_names[0]}, {shape_str})"
