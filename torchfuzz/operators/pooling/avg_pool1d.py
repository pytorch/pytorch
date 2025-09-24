"""1D average pooling operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class AvgPool1dOperator(Operator):
    """Operator for 1D average pooling (torch.nn.functional.avg_pool1d)."""

    def __init__(self):
        super().__init__("avg_pool1d")

    def can_produce(self, tensor):
        """AvgPool1d can produce 3D tensors (batch, channels, length)."""
        return len(tensor.size) == 3

    def decompose(self, tensor):
        """Decompose tensor into input tensors for avg_pool1d operation."""
        # tensor shape is (batch_size, channels, out_length)
        batch_size, channels, out_length = tensor.size

        # Choose pooling parameters
        kernel_size = random.choice([1, 2, 3])  # Small kernel sizes
        stride = kernel_size  # Use stride=kernel_size for simplicity (no overlap)

        # Calculate input length that will produce exact output length
        # For avg_pool: out_length = floor((in_length - kernel_size) / stride) + 1
        # So: in_length = (out_length - 1) * stride + kernel_size
        in_length = (out_length - 1) * stride + kernel_size
        in_length = max(in_length, kernel_size)  # Ensure valid input size

        # Verify the calculation works
        calculated_out_length = (in_length - kernel_size) // stride + 1
        if calculated_out_length != out_length:
            # If it doesn't work, use kernel_size=1 and stride=1 for guaranteed match
            kernel_size = 1
            stride = 1
            in_length = out_length  # For kernel=1, stride=1: out_length = in_length

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

        # Store parameters for codegen
        self._kernel_size = kernel_size
        self._stride = stride
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for avg_pool1d operation."""
        kernel_size = getattr(self, '_kernel_size', 2)
        stride = getattr(self, '_stride', 2)

        return f"{output_name} = torch.nn.functional.avg_pool1d({input_names[0]}, kernel_size={kernel_size}, stride={stride})"
