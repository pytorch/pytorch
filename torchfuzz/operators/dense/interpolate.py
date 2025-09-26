"""Interpolate operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class InterpolateOperator(Operator):
    """Operator for interpolation (torch.nn.functional.interpolate)."""

    def __init__(self):
        super().__init__("interpolate")

    def can_produce(self, tensor):
        """Interpolate can produce 3D-5D tensors and only for floating point dtypes."""
        allowed_dtypes = {"float32", "float64", "bfloat16", "float16"}
        return 3 <= len(tensor.size) <= 5 and str(tensor.dtype).lower() in allowed_dtypes

    def decompose(self, tensor):
        """Decompose tensor into input tensor for interpolate operation."""
        # tensor shape depends on input dimensionality:
        # 3D: (N, C, L) - batch_size, channels, length
        # 4D: (N, C, H, W) - batch_size, channels, height, width
        # 5D: (N, C, D, H, W) - batch_size, channels, depth, height, width

        if len(tensor.size) < 3:
            raise ValueError("Interpolate requires at least 3D input")

        batch_size, channels = tensor.size[:2]
        output_spatial_dims = tensor.size[2:]

        # Choose random input spatial dimensions (smaller than output for upsampling)
        input_spatial_dims = []
        for out_dim in output_spatial_dims:
            # Make input smaller for upsampling (common case)
            scale_factor = random.choice([0.5, 0.25, 0.125])
            in_dim = max(1, int(out_dim * scale_factor))
            input_spatial_dims.append(in_dim)

        input_size = (batch_size, channels) + tuple(input_spatial_dims)
        input_stride = self._calc_stride(input_size)

        # Create input tensor
        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        result = [input_tensor]

        # Store parameters for codegen
        self._size = output_spatial_dims
        self._mode = random.choice(['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic'])

        # Ensure mode is compatible with dimensionality
        if len(output_spatial_dims) == 1:
            self._mode = random.choice(['nearest', 'linear'])
        elif len(output_spatial_dims) == 2:
            self._mode = random.choice(['nearest', 'bilinear', 'bicubic'])
        elif len(output_spatial_dims) == 3:
            self._mode = random.choice(['nearest', 'trilinear'])

        return result

    def _calc_stride(self, size):
        """Calculate stride for contiguous tensor."""
        stride = [1]
        for dim in reversed(size[:-1]):
            stride.insert(0, stride[0] * dim)
        return tuple(stride)

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for interpolate operation."""
        size = getattr(self, '_size', None)
        mode = getattr(self, '_mode', 'nearest')

        size_str = str(size) if size else 'None'
        return f"{output_name} = torch.nn.functional.interpolate({input_names[0]}, size={size_str}, mode='{mode}')"
