"""3D max pooling operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MaxPool3dOperator(Operator):
    """Operator for 3D max pooling (torch.nn.functional.max_pool3d)."""

    def __init__(self):
        super().__init__("max_pool3d")

    def can_produce(self, tensor):
        """MaxPool3d can produce 5D tensors (batch, channels, depth, height, width)."""
        return len(tensor.size) == 5

    def decompose(self, tensor):
        """Decompose tensor into input tensors for max_pool3d operation."""
        # tensor shape is (batch_size, channels, out_depth, out_height, out_width)
        batch_size, channels, out_depth, out_height, out_width = tensor.size

        # Choose pooling parameters
        kernel_size = random.choice([1, 2])  # Small kernel sizes for 3D to avoid memory issues
        stride = kernel_size  # Use stride=kernel_size for simplicity (no overlap)

        # Calculate input dimensions that will produce exact output dimensions
        # For max_pool: out_dim = floor((in_dim - kernel_size) / stride) + 1
        # So: in_dim = (out_dim - 1) * stride + kernel_size
        in_depth = (out_depth - 1) * stride + kernel_size
        in_height = (out_height - 1) * stride + kernel_size
        in_width = (out_width - 1) * stride + kernel_size
        in_depth = max(in_depth, kernel_size)  # Ensure valid input size
        in_height = max(in_height, kernel_size)
        in_width = max(in_width, kernel_size)

        # Verify the calculation works
        calculated_out_depth = (in_depth - kernel_size) // stride + 1
        calculated_out_height = (in_height - kernel_size) // stride + 1
        calculated_out_width = (in_width - kernel_size) // stride + 1
        if (calculated_out_depth != out_depth or
            calculated_out_height != out_height or
            calculated_out_width != out_width):
            # If it doesn't work, use kernel_size=1 and stride=1 for guaranteed match
            kernel_size = 1
            stride = 1
            in_depth = out_depth    # For kernel=1, stride=1: out_dim = in_dim
            in_height = out_height
            in_width = out_width

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

        # Store parameters for codegen
        self._kernel_size = kernel_size
        self._stride = stride
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for max_pool3d operation."""
        kernel_size = getattr(self, '_kernel_size', 2)
        stride = getattr(self, '_stride', 2)

        return f"{output_name} = torch.nn.functional.max_pool3d({input_names[0]}, kernel_size={kernel_size}, stride={stride})"
