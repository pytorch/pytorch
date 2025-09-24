"""Random normal like operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RandnLikeOperator(Operator):
    """Operator for torch.randn_like."""

    def __init__(self):
        super().__init__("randn_like")

    def can_produce(self, tensor):
        """torch.randn_like can produce any floating point tensor."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.randn_like needs an input tensor with any dtype."""
        input_dtype = "float32"

        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(tensor.size)
        input_tensor = Tensor(tensor.size, input_stride, input_dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.randn_like operation."""
        if len(input_names) != 1:
            raise ValueError("torch.randn_like requires exactly 1 input")

        dtype_str = f"dtype=torch.{output_tensor.dtype}"
        return f"{output_name} = torch.randn_like({input_names[0]}, {dtype_str})"
