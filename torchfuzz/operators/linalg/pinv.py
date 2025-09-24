"""Linear algebra pseudo-inverse operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class PinvOperator(Operator):
    """Operator for torch.linalg.pinv."""

    def __init__(self):
        super().__init__("linalg.pinv")

    def can_produce(self, tensor):
        """linalg.pinv can produce matrices from floating point matrix inputs."""
        # linalg.pinv only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Output must be at least 2D
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input tensor for pseudo-inverse."""
        # For pseudo-inverse, output shape is transposed of input shape
        # If input is (..., m, n), output is (..., n, m)
        batch_dims = tensor.size[:-2]
        m, n = tensor.size[-2], tensor.size[-1]
        input_size = batch_dims + (n, m)  # Transpose the last two dimensions

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
        """Generate code for linalg.pinv operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.pinv requires exactly 1 input")

        return f"{output_name} = torch.linalg.pinv({input_names[0]})"
