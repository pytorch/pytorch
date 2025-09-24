"""Linear algebra Cholesky decomposition operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class CholeskyOperator(Operator):
    """Operator for torch.linalg.cholesky."""

    def __init__(self):
        super().__init__("linalg.cholesky")

    def can_produce(self, tensor):
        """linalg.cholesky can produce square matrices from floating point positive definite matrices."""
        # linalg.cholesky only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Output must be at least 2D and square matrices
        if len(tensor.size) < 2:
            return False
        # Last two dimensions must be equal (square matrix)
        return tensor.size[-1] == tensor.size[-2]

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Cholesky decomposition."""
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
        """Generate code for linalg.cholesky operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.cholesky requires exactly 1 input")

        return f"{output_name} = torch.linalg.cholesky({input_names[0]})"
