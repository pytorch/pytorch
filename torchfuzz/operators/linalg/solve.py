"""Linear algebra solve operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class SolveOperator(Operator):
    """Operator for torch.linalg.solve."""

    def __init__(self):
        super().__init__("linalg.solve")

    def can_produce(self, tensor):
        """linalg.solve can produce solutions from floating point system Ax=B."""
        # linalg.solve only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Output must be at least 1D
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensors for linear system solve."""
        # For solve(A, B) -> X, where A @ X = B
        # If output X has shape (..., n, k), then A has shape (..., n, n) and B has shape (..., n, k)

        if len(tensor.size) == 1:
            # X is 1D with shape (n,), A is (n, n), B is (n,)
            n = tensor.size[0]
            A_size = (n, n)
            B_size = (n,)
        else:
            # X has shape (..., n, k), A has shape (..., n, n), B has shape (..., n, k)
            *batch_dims, n, k = tensor.size
            A_size = tuple(batch_dims) + (n, n)
            B_size = tensor.size

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        A_stride = calc_stride(A_size)
        B_stride = calc_stride(B_size)

        A_tensor = Tensor(A_size, A_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        B_tensor = Tensor(B_size, B_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        return [A_tensor, B_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for linalg.solve operation."""
        if len(input_names) != 2:
            raise ValueError("linalg.solve requires exactly 2 inputs (A, B)")

        return f"{output_name} = torch.linalg.solve({input_names[0]}, {input_names[1]})"
