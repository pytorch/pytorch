"""GLU (Gated Linear Unit) operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class GluOperator(Operator):
    """Operator for GLU (Gated Linear Unit) activation function (torch.nn.functional.glu)."""

    def __init__(self):
        super().__init__("glu")

    def can_produce(self, tensor):
        """GLU can produce tensors where the input dimension is even (to split in half)."""
        # GLU needs at least 1 dimension and the size along the gating dimension must be even
        if len(tensor.size) < 1:
            return False

        # Check if any dimension has even size > 1 (needed for GLU to split)
        for dim_size in tensor.size:
            if dim_size > 1 and dim_size % 2 == 0:
                return True
        return False

    def decompose(self, tensor):
        """Decompose tensor into input tensor for GLU."""
        # GLU splits the input along a dimension, so the input dimension should be 2x the output
        output_size = list(tensor.size)

        # Choose a dimension to apply GLU on (must have even size in output)
        possible_dims = []
        for i, dim_size in enumerate(output_size):
            if dim_size > 0:  # Any dimension with positive size can be used
                possible_dims.append(i)

        if not possible_dims:
            # Fallback to last dimension
            dim = -1
        else:
            dim = random.choice(possible_dims)

        # Input size should be 2x the output size along the chosen dimension
        input_size = output_size.copy()
        input_size[dim] = output_size[dim] * 2

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim_val in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim_val)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        input_tensor = Tensor(tuple(input_size), input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Store the dimension for codegen
        self._dim = dim

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for GLU operation."""
        dim = getattr(self, '_dim', -1)  # Default to last dimension
        return f"{output_name} = torch.nn.functional.glu({input_names[0]}, dim={dim})"
