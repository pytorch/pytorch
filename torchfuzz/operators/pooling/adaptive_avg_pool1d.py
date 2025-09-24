"""1D adaptive average pooling operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class AdaptiveAvgPool1dOperator(Operator):
    """Operator for 1D adaptive average pooling (torch.nn.functional.adaptive_avg_pool1d)."""

    def __init__(self):
        super().__init__("adaptive_avg_pool1d")

    def can_produce(self, tensor):
        """AdaptiveAvgPool1d can produce 3D tensors (batch, channels, length)."""
        return len(tensor.size) == 3

    def decompose(self, tensor):
        """Decompose tensor into input tensors for adaptive_avg_pool1d operation."""
        # tensor shape is (batch_size, channels, out_length)
        batch_size, channels, out_length = tensor.size

        # For adaptive pooling, we need a larger input than output
        # Choose an input length that's reasonable
        in_length = max(out_length, random.choice([16, 32, 64, 128]))

        # Input tensor: (batch_size, channels, in_length)
        input_size = (batch_size, channels, in_length)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        # Type promotion for realistic types
        dtype = tensor.dtype

        # Create input tensor
        input_tensor = Tensor(input_size, input_stride, dtype, tensor.device, tensor.supported_ops)

        result = [input_tensor]

        # Store output size for codegen
        self._output_size = out_length
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for adaptive_avg_pool1d operation."""
        output_size = getattr(self, '_output_size', 1)

        return f"{output_name} = torch.nn.functional.adaptive_avg_pool1d({input_names[0]}, {output_size})"
