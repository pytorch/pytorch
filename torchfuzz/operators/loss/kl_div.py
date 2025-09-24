"""KL divergence loss operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class KlDivOperator(Operator):
    """Operator for torch.nn.functional.kl_div."""

    def __init__(self):
        super().__init__("nn.functional.kl_div")

    def can_produce(self, tensor):
        """kl_div can produce scalar or reduced tensors from floating point inputs."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """kl_div needs input and target tensors with same shape."""
        if len(tensor.size) == 0:
            input_size = (4, 4)
        else:
            input_size = tensor.size

        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        target_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        return [input_tensor, target_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for kl_div operation."""
        if len(input_names) != 2:
            raise ValueError("kl_div requires exactly 2 inputs (input, target)")

        if len(output_tensor.size) == 0:
            return f"{output_name} = torch.nn.functional.kl_div({input_names[0]}, {input_names[1]})"
        else:
            return f"{output_name} = torch.nn.functional.kl_div({input_names[0]}, {input_names[1]}, reduction='none')"
