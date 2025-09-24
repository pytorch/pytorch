"""Linear algebra determinant operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class DetOperator(Operator):
    """Operator for torch.linalg.det."""

    def __init__(self):
        super().__init__("linalg.det")

    def can_produce(self, tensor):
        """linalg.det can produce scalars from floating point square matrix inputs."""
        # linalg.det only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # det reduces square matrices to scalars (or batch of scalars)
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for determinant calculation."""
        # The input must be a square matrix with possibly batch dimensions
        # If output is scalar, input is (..., n, n)
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
        """Generate code for linalg.det operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.det requires exactly 1 input")

        return f"{output_name} = torch.linalg.det({input_names[0]})"
