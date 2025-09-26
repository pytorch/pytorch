"""2D max pooling operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MaxPool2dOperator(Operator):
    """Operator for 2D max pooling (torch.max_pool2d)."""

    def __init__(self):
        super().__init__("max_pool2d")

    def can_produce(self, tensor):
        """MaxPool2d can produce 4D tensors (batch, channels, height, width) and only for floating point dtypes."""
        allowed_dtypes = {"float32", "float64", "bfloat16", "float16"}
        return len(tensor.size) == 4 and str(tensor.dtype).lower() in allowed_dtypes

    def decompose(self, tensor):
        """Decompose tensor into input tensor for max_pool2d operation."""
        # tensor shape is (batch_size, channels, out_height, out_width)
        batch_size, channels, out_height, out_width = tensor.size

        # Choose pooling parameters
        kernel_size = random.choice([2, 3])  # Common pooling kernel sizes
        stride = kernel_size  # Common pattern: stride = kernel_size
        padding = 0  # Typically no padding for pooling

        # Calculate input dimensions that will produce exact output dimensions
        # For pooling: out_dim = (in_dim + 2 * padding - kernel_size) / stride + 1
        # Solve for in_dim: in_dim = (out_dim - 1) * stride + kernel_size - 2 * padding
        in_height = (out_height - 1) * stride + kernel_size - 2 * padding
        in_width = (out_width - 1) * stride + kernel_size - 2 * padding

        # Ensure valid input size (at least as large as kernel)
        in_height = max(in_height, kernel_size)
        in_width = max(in_width, kernel_size)

        # Verify the calculation works for both dimensions
        calculated_out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        calculated_out_width = (in_width + 2 * padding - kernel_size) // stride + 1

        if calculated_out_height != out_height or calculated_out_width != out_width:
            # If it doesn't work, use kernel_size=stride=1 for guaranteed match
            kernel_size = 1
            stride = 1
            padding = 0
            in_height = out_height
            in_width = out_width

        # Input tensor: (batch_size, channels, in_height, in_width)
        input_size = (batch_size, channels, in_height, in_width)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        # Create input tensor
        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        result = [input_tensor]

        # Store parameters for codegen
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for max_pool2d operation."""
        kernel_size = getattr(self, '_kernel_size', 2)
        stride = getattr(self, '_stride', 2)
        padding = getattr(self, '_padding', 0)

        return f"{output_name} = torch.max_pool2d({input_names[0]}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
