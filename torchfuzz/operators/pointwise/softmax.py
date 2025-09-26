"""Softmax operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class SoftmaxOperator(Operator):
    """Operator for softmax (torch.softmax)."""

    def __init__(self):
        super().__init__("softmax")

    def can_produce(self, tensor):
        """Softmax can be applied to any tensor with floating point dtypes."""
        allowed_dtypes = {"float32", "float64", "bfloat16", "float16"}
        return str(tensor.dtype).lower() in allowed_dtypes

    def decompose(self, tensor):
        """Decompose tensor into input tensor for softmax."""
        # The input to softmax must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for softmax operation."""
        # Choose a random dimension for softmax
        num_dims = len(output_tensor.size)
        if num_dims == 0:
            # Scalar tensor - softmax with dim=0 works for scalars
            dim = 0
        else:
            dim = random.choice(range(num_dims))
        return f"{output_name} = torch.softmax({input_names[0]}, dim={dim})"
