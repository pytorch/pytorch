"""Add matrix multiplication operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class AddmmOperator(Operator):
    """Operator for addmm (torch.addmm): bias + mat1 @ mat2."""

    def __init__(self):
        super().__init__("addmm")

    def can_produce(self, tensor):
        """Addmm can produce tensors that are 2D."""
        return len(tensor.size) == 2

    def decompose(self, tensor, num_inputs=3):
        """Decompose tensor into input tensors for addmm."""
        if num_inputs != 3:
            raise ValueError("Addmm requires exactly 3 inputs (bias, mat1, mat2)")

        # tensor is (m, n), we need bias (m, n), mat1 (m, k), mat2 (k, n)
        m, n = tensor.size
        # Choose a random inner dimension k
        k = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

        # No type promotion: use the same dtype for all inputs
        input_dtypes = (tensor.dtype, tensor.dtype, tensor.dtype)

        # Create input tensors: bias (m, n), mat1 (m, k), mat2 (k, n)
        bias = Tensor((m, n), (n, 1), input_dtypes[0], tensor.device, tensor.supported_ops)
        mat1 = Tensor((m, k), (k, 1), input_dtypes[1], tensor.device, tensor.supported_ops)
        mat2 = Tensor((k, n), (n, 1), input_dtypes[2], tensor.device, tensor.supported_ops)

        return [bias, mat1, mat2]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for addmm operation."""
        if len(input_names) != 3:
            raise ValueError("Addmm requires exactly 3 inputs (bias, mat1, mat2)")
        return f"{output_name} = torch.addmm({input_names[0]}, {input_names[1]}, {input_names[2]})"
