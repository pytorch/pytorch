"""Matrix multiplication operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MmOperator(Operator):
    """Operator for matrix multiplication (torch.mm)."""

    def __init__(self):
        super().__init__("mm")

    def can_produce(self, tensor):
        """MM can produce tensors that are 2D and floating point."""
        # mm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) == 2

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for matrix multiplication."""
        if num_inputs != 2:
            raise ValueError("Matrix multiplication requires exactly 2 inputs")

        # tensor is (m, n), we need (m, k) @ (k, n)
        m, n = tensor.size
        # Choose a random inner dimension k
        k = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

        # Type promotion table for realistic LLM/diffusion model types
        input_dtypes = (tensor.dtype, tensor.dtype)

        # Create input tensors: (m, k) and (k, n)
        input1 = Tensor((m, k), (k, 1), input_dtypes[0], tensor.device, tensor.supported_ops)
        input2 = Tensor((k, n), (n, 1), input_dtypes[1], tensor.device, tensor.supported_ops)

        return [input1, input2]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for matrix multiplication operation."""
        if len(input_names) != 2:
            raise ValueError("Matrix multiplication requires exactly 2 inputs")
        return f"{output_name} = torch.mm({input_names[0]}, {input_names[1]})"
