"""Trace operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class TraceOperator(Operator):
    """Operator for matrix trace (sum of diagonal elements)."""

    def __init__(self):
        super().__init__("trace")

    def can_produce(self, tensor):
        """Trace can produce scalars or batched scalars from floating point square matrices."""
        # Trace only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Trace reduces square matrices to scalars (or batch of scalars)
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for trace calculation."""
        # The input must be a square matrix with possibly batch dimensions
        # If output is scalar, input is (n, n)
        # If output has batch dims, input has same batch dims plus (n, n)

        # Add square matrix dimensions to the output shape
        n = 4  # Choose a reasonable matrix size
        input_size = tensor.size + (n, n)

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
        """Generate code for trace operation."""
        if len(input_names) != 1:
            raise ValueError("trace requires exactly 1 input")

        return f"{output_name} = torch.trace({input_names[0]})"
