"""Normal distribution operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class NormalOperator(Operator):
    """Operator for torch.normal."""

    def __init__(self):
        super().__init__("normal")

    def can_produce(self, tensor):
        """torch.normal can produce any floating point tensor."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.normal needs mean and std tensors."""
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(tensor.size)
        mean_tensor = Tensor(tensor.size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        std_tensor = Tensor(tensor.size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [mean_tensor, std_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.normal operation."""
        if len(input_names) != 2:
            raise ValueError("torch.normal requires exactly 2 inputs (mean, std)")

        return f"{output_name} = torch.normal({input_names[0]}, {input_names[1]})"
