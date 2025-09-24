"""2D transposed convolution operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ConvTranspose2dOperator(Operator):
    """Operator for 2D transposed convolution (torch.nn.functional.conv_transpose2d)."""

    def __init__(self):
        super().__init__("conv_transpose2d")

    def can_produce(self, tensor):
        """ConvTranspose2d can produce 4D tensors (batch, out_channels, height, width)."""
        return len(tensor.size) == 4

    def decompose(self, tensor):
        """Decompose tensor into input tensors for conv_transpose2d operation."""
        # tensor shape is (batch_size, out_channels, out_height, out_width)
        batch_size, out_channels, out_height, out_width = tensor.size

        # Choose input parameters to ensure exact output dimensions
        in_channels = random.choice([64, 128, 256, 512, 768, 1024])
        kernel_size = random.choice([1, 3, 5])  # Use smaller kernels to avoid issues
        stride = 1  # Use stride=1 for simplicity
        padding = 0  # Use padding=0 for simplicity

        # For transposed convolution with stride=1, padding=0:
        # out_dim = in_dim + kernel_size - 1
        # So: in_dim = out_dim - kernel_size + 1
        in_height = out_height - kernel_size + 1
        in_width = out_width - kernel_size + 1
        in_height = max(in_height, 1)  # Ensure valid input size
        in_width = max(in_width, 1)

        # Verify the calculation works
        calculated_out_height = in_height + kernel_size - 1
        calculated_out_width = in_width + kernel_size - 1
        if calculated_out_height != out_height or calculated_out_width != out_width:
            # If it doesn't work, use kernel_size=1 for guaranteed match
            kernel_size = 1
            in_height = out_height  # For kernel=1: out_dim = in_dim
            in_width = out_width

        # Input tensor: (batch_size, in_channels, in_height, in_width)
        input_size = (batch_size, in_channels, in_height, in_width)

        # Weight tensor: (in_channels, out_channels, kernel_size, kernel_size)
        weight_size = (in_channels, out_channels, kernel_size, kernel_size)

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
        """Generate code for conv_transpose2d operation."""
        stride = getattr(self, '_stride', 1)
        padding = getattr(self, '_padding', 0)

        return f"{output_name} = torch.nn.functional.conv_transpose2d({input_names[0]}, {input_names[1]}, stride={stride}, padding={padding})"
