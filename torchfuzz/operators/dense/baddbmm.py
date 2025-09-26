"""Batch add matrix multiplication operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class BaddbmmOperator(Operator):
    """Operator for baddbmm (torch.baddbmm): bias + batch1 @ batch2."""

    def __init__(self):
        super().__init__("baddbmm")

    def can_produce(self, tensor):
        """Baddbmm can produce tensors that are 3D and floating point."""
        # baddbmm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) == 3

    def decompose(self, tensor, num_inputs=3):
        """Decompose tensor into input tensors for baddbmm."""
        if num_inputs != 3:
            raise ValueError("Baddbmm requires exactly 3 inputs (bias, batch1, batch2)")

        # tensor is (b, m, n), we need bias (b, m, n), batch1 (b, m, k), batch2 (b, k, n)
        b, m, n = tensor.size
        # Choose a random inner dimension k
        k = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

        input_dtypes = (tensor.dtype, tensor.dtype, tensor.dtype)

        # Create input tensors: bias (b, m, n), batch1 (b, m, k), batch2 (b, k, n)
        bias = Tensor((b, m, n), (m * n, n, 1), input_dtypes[0], tensor.device, tensor.supported_ops)
        batch1 = Tensor((b, m, k), (m * k, k, 1), input_dtypes[1], tensor.device, tensor.supported_ops)
        batch2 = Tensor((b, k, n), (k * n, n, 1), input_dtypes[2], tensor.device, tensor.supported_ops)

        return [bias, batch1, batch2]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for baddbmm operation."""
        if len(input_names) != 3:
            raise ValueError("Baddbmm requires exactly 3 inputs (bias, batch1, batch2)")
        return f"{output_name} = torch.baddbmm({input_names[0]}, {input_names[1]}, {input_names[2]})"
