"""Random integer like operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RandintLikeOperator(Operator):
    """Operator for torch.randint_like."""

    def __init__(self):
        super().__init__("randint_like")

    def can_produce(self, tensor):
        """torch.randint_like can produce any integer tensor."""
        return tensor.dtype in ["int8", "int16", "int32", "int64", "uint8"]

    def decompose(self, tensor):
        """torch.randint_like needs an input tensor with any dtype."""
        input_dtype = "int32"

        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(tensor.size)
        input_tensor = Tensor(tensor.size, input_stride, input_dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.randint_like operation."""
        if len(input_names) != 1:
            raise ValueError("torch.randint_like requires exactly 1 input")

        dtype_str = f"dtype=torch.{output_tensor.dtype}"
        return f"{output_name} = torch.randint_like({input_names[0]}, 0, 10, {dtype_str})"
