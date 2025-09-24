"""Linear algebra matrix inverse operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class InvOperator(Operator):
    """Operator for torch.linalg.inv."""

    def __init__(self):
        super().__init__("linalg.inv")

    def can_produce(self, tensor):
        """linalg.inv can produce square matrices from floating point square matrix inputs."""
        # linalg.inv only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Output must be at least 2D and square matrices
        if len(tensor.size) < 2:
            return False
        # Last two dimensions must be equal (square matrix)
        return tensor.size[-1] == tensor.size[-2]

    def decompose(self, tensor):
        """Decompose tensor into input tensor for matrix inverse."""
        # Input must have the same shape as output (square matrix)
        input_size = tensor.size

        # Calculate stride for contiguous tensor
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for linalg.inv operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.inv requires exactly 1 input")

        return f"{output_name} = torch.linalg.inv({input_names[0]})"
