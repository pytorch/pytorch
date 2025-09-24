"""Matrix multiplication with broadcasting operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MatmulOperator(Operator):
    """Operator for matrix multiplication with broadcasting (torch.matmul)."""

    def __init__(self):
        super().__init__("matmul")

    def can_produce(self, tensor):
        """Matmul can produce tensors that are at least 1D and floating point."""
        # matmul only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # matmul requires at least 1D tensors
        return len(tensor.size) >= 1

    def decompose(self, tensor):
        """Decompose tensor into input tensors for matrix multiplication."""
        ndim = len(tensor.size)

        if ndim == 1:
            # Output is 1D: this can come from (1,n) @ (n,1) -> (1,1) -> squeezed to (1,)
            # But torch.matmul with 1D output actually comes from (n,) @ (n,m) -> (m,) or (m,n) @ (n,) -> (m,)
            # Let's implement the second case: (m,n) @ (n,) -> (m,)
            m = tensor.size[0]
            n = random.choice([64, 128, 256, 512, 768, 1024])

            input1_size = (m, n)
            input2_size = (n,)

            input1_stride = (n, 1)
            input2_stride = (1,)

        elif ndim == 2:
            # Output is 2D: (m, n) can come from (m, k) @ (k, n)
            m, n = tensor.size
            k = random.choice([64, 128, 256, 512, 768, 1024])

            input1_size = (m, k)
            input2_size = (k, n)

            input1_stride = (k, 1)
            input2_stride = (n, 1)

        else:
            # For higher dimensions, matmul applies to the last two dimensions
            # and broadcasts over the batch dimensions
            *batch_dims, m, n = tensor.size
            k = random.choice([64, 128, 256, 512, 768, 1024])

            input1_size = tuple(batch_dims + [m, k])
            input2_size = tuple(batch_dims + [k, n])

            # Calculate strides for contiguous tensors
            def calc_stride(size):
                stride = [1]
                for dim in reversed(size[:-1]):
                    stride.insert(0, stride[0] * dim)
                return tuple(stride)

            input1_stride = calc_stride(input1_size)
            input2_stride = calc_stride(input2_size)

        # Type promotion for realistic types
        dtype = tensor.dtype

        # Create input tensors
        input1_tensor = Tensor(input1_size, input1_stride, dtype, tensor.device, tensor.supported_ops)
        input2_tensor = Tensor(input2_size, input2_stride, dtype, tensor.device, tensor.supported_ops)

        return [input1_tensor, input2_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for matmul operation."""
        if len(input_names) != 2:
            raise ValueError("Matmul requires exactly 2 inputs")
        return f"{output_name} = torch.matmul({input_names[0]}, {input_names[1]})"
