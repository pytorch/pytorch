"""Bernoulli distribution operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class BernoulliOperator(Operator):
    """Operator for torch.bernoulli."""

    def __init__(self):
        super().__init__("bernoulli")

    def can_produce(self, tensor):
        """torch.bernoulli can produce any floating point tensor."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.bernoulli needs an input probability tensor."""
        # Input should be same shape as output, containing probabilities
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(tensor.size)
        input_tensor = Tensor(tensor.size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.bernoulli operation."""
        if len(input_names) != 1:
            raise ValueError("torch.bernoulli requires exactly 1 input")

        return f"{output_name} = torch.bernoulli({input_names[0]})"
