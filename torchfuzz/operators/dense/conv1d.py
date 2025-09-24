"""1D convolution operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class Conv1dOperator(Operator):
    """Operator for 1D convolution (torch.nn.functional.conv1d)."""

    def __init__(self):
        super().__init__("conv1d")

    def can_produce(self, tensor):
        """Conv1d can produce 3D tensors (batch, out_channels, length)."""
        return len(tensor.size) == 3

    def decompose(self, tensor):
        """Decompose tensor into input tensors for conv1d operation."""
        # tensor shape is (batch_size, out_channels, out_length)
        batch_size, out_channels, out_length = tensor.size

        # Choose input parameters to ensure exact output dimensions
        in_channels = random.choice([64, 128, 256, 512, 768, 1024])
        kernel_size = random.choice([1, 3, 5])  # Use smaller kernels to avoid issues
        stride = 1  # Use stride=1 for simplicity

        # Calculate input length that will produce exact output length
        # For stride=1: out_length = in_length + 2 * padding - kernel_size + 1
        # We'll choose padding=0 for simplicity and calculate in_length
        padding = 0

        # Solve for in_length: in_length = out_length - 2 * padding + kernel_size - 1
        in_length = out_length - 2 * padding + kernel_size - 1
        in_length = max(in_length, kernel_size)  # Ensure valid input size

        # Verify the calculation works
        calculated_out_length = (in_length + 2 * padding - kernel_size) // stride + 1
        if calculated_out_length != out_length:
            # If it doesn't work, use kernel_size=1 and padding=0 for guaranteed match
            kernel_size = 1
            padding = 0
            in_length = out_length  # For kernel=1, stride=1, padding=0: out_length = in_length

        # Input tensor: (batch_size, in_channels, in_length)
        input_size = (batch_size, in_channels, in_length)

        # Weight tensor: (out_channels, in_channels, kernel_size)
        weight_size = (out_channels, in_channels, kernel_size)

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        weight_stride = calc_stride(weight_size)

        # Type promotion for realistic types
        dtype = tensor.dtype

        # Create input tensors
        input_tensor = Tensor(input_size, input_stride, dtype, tensor.device, tensor.supported_ops)
        weight_tensor = Tensor(weight_size, weight_stride, dtype, tensor.device, tensor.supported_ops)

        result = [input_tensor, weight_tensor]

        # Store parameters for codegen
        self._stride = stride
        self._padding = padding
        return result

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for conv1d operation."""
        stride = getattr(self, '_stride', 1)
        padding = getattr(self, '_padding', 0)

        return f"{output_name} = torch.nn.functional.conv1d({input_names[0]}, {input_names[1]}, stride={stride}, padding={padding})"
