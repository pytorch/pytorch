"""Argsort operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ArgsortOperator(Operator):
    """Operator for torch.argsort returning indices along a dimension."""

    def __init__(self):
        super().__init__("argsort")

    def can_produce(self, tensor):
        """Argsort returns indices; output dtype must be int64."""
        return tensor.dtype == "int64" and len(tensor.size) >= 1

    def decompose(self, tensor):
        """Create an input tensor of same shape; store chosen dim and order."""
        # Choose a valid dimension to sort over
        dim = random.randrange(len(tensor.size))
        descending = random.choice([True, False])
        # Store parameters on the output tensor for codegen
        tensor._argsort_dim = dim
        tensor._argsort_desc = descending

        # Create a contiguous input tensor with a floating dtype
        input_stride = self._calc_stride(tensor.size)
        input_dtype = random.choice(["float32", "float64", "bfloat16", "float16"])
        input_tensor = Tensor(tensor.size, input_stride, input_dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def _calc_stride(self, size):
        stride = [1]
        for dim in reversed(size[:-1]):
            stride.insert(0, stride[0] * dim)
        return tuple(stride)

    def codegen(self, output_name, input_names, output_tensor):
        dim = getattr(output_tensor, "_argsort_dim", 0)
        desc = getattr(output_tensor, "_argsort_desc", False)
        return f"{output_name} = torch.argsort({input_names[0]}, dim={dim}, descending={desc})"
