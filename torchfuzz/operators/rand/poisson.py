"""Poisson distribution operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class PoissonOperator(Operator):
    """Operator for torch.poisson."""

    def __init__(self):
        super().__init__("poisson")

    def can_produce(self, tensor):
        """torch.poisson can produce any floating point tensor."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.poisson needs a rate tensor."""
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(tensor.size)
        rate_tensor = Tensor(tensor.size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [rate_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.poisson operation."""
        if len(input_names) != 1:
            raise ValueError("torch.poisson requires exactly 1 input")

        return f"{output_name} = torch.poisson({input_names[0]})"
