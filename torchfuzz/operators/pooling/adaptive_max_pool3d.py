"""3D adaptive max pooling operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class AdaptiveMaxPool3dOperator(Operator):
    """Operator for 3D adaptive max pooling (torch.nn.functional.adaptive_max_pool3d)."""

    def __init__(self):
        super().__init__("adaptive_max_pool3d")

    def can_produce(self, tensor):
        """AdaptiveMaxPool3d can produce 5D tensors (batch, channels, depth, height, width)."""
        return len(tensor.size) == 5

    def decompose(self, tensor):
        """Decompose tensor into input tensors for adaptive_max_pool3d operation."""
        # tensor shape is (batch_size, channels, out_depth, out_height, out_width)
        batch_size, channels, out_depth, out_height, out_width = tensor.size

        # For adaptive pooling, we need a larger input than output
        # Choose input dimensions that are reasonable
        in_depth = max(out_depth, random.choice([8, 16, 32, 64]))
        in_height = max(out_height, random.choice([8, 16, 32, 64]))
        in_width = max(out_width, random.choice([8, 16, 32, 64]))

        # Input tensor: (batch_size, channels, in_depth, in_height, in_width)
        input_size = (batch_size, channels, in_depth, in_height, in_width)

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
        self._output_size = (out_depth, out_height, out_width)
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for adaptive_max_pool3d operation."""
        output_size = getattr(self, '_output_size', (1, 1, 1))

        return f"{output_name} = torch.nn.functional.adaptive_max_pool3d({input_names[0]}, {output_size})"
