"""RMS normalization operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class RmsNormOperator(Operator):
    """Operator for RMS normalization (torch.nn.functional.rms_norm)."""

    def __init__(self):
        super().__init__("rms_norm")

    def can_produce(self, tensor):
        """RmsNorm can produce tensors of various dimensions with floating point types."""
        # RMS norm only supports floating point dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # RMS norm needs at least 1 dimension to work on
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensor for RMS norm operation."""
        # The input to RMS norm must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # normalized_shape is typically the last few dimensions
        # Similar to layer norm patterns
        ndim = len(tensor.size)
        if ndim >= 3:
            # For 3D+ tensors, typically normalize over last 1 or 2 dimensions
            norm_dims = random.choice([1, 2])
        else:
            # For lower dimensional tensors, normalize over last dimension
            norm_dims = 1

        normalized_shape = tensor.size[-norm_dims:]

        # Store normalized_shape for codegen
        self._normalized_shape = normalized_shape

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for RMS norm operation."""
        normalized_shape = getattr(self, '_normalized_shape', output_tensor.size[-1:])

        # Validate that normalized_shape matches the last dimensions of the input tensor
        input_shape = output_tensor.size
        if len(normalized_shape) > len(input_shape) or normalized_shape != input_shape[-len(normalized_shape):]:
            # Fallback: use the last dimension as normalized_shape
            normalized_shape = (input_shape[-1],)

        shape_str = str(normalized_shape)
        return f"{output_name} = torch.nn.functional.rms_norm({input_names[0]}, {shape_str})"
