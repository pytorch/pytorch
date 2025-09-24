"""Batch matrix multiplication operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class BmmOperator(Operator):
    """Operator for batch matrix multiplication (torch.bmm)."""

    def __init__(self):
        super().__init__("bmm")

    def can_produce(self, tensor, max_numel=1_000_000):
        """BMM can produce tensors that are 3D, floating point, and not too large."""
        # bmm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) == 3 and (tensor.size[0] * tensor.size[1] * tensor.size[2] <= max_numel)

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for batch matrix multiplication."""
        if num_inputs != 2:
            raise ValueError("Batch matrix multiplication requires exactly 2 inputs")

        # tensor is (b, m, n), we need (b, m, k) @ (b, k, n)
        b, m, n = tensor.size
        # Choose a random inner dimension k
        k = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

        dtype = tensor.dtype
        input_dtypes = (dtype, dtype)

        # Create input tensors: (b, m, k) and (b, k, n)
        input1 = Tensor((b, m, k), (m * k, k, 1), input_dtypes[0], tensor.device, tensor.supported_ops)
        input2 = Tensor((b, k, n), (k * n, n, 1), input_dtypes[1], tensor.device, tensor.supported_ops)

        return [input1, input2]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for batch matrix multiplication operation."""
        if len(input_names) != 2:
            raise ValueError("Batch matrix multiplication requires exactly 2 inputs")
        return f"{output_name} = torch.bmm({input_names[0]}, {input_names[1]})"
