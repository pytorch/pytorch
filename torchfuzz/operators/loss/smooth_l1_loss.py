"""Smooth L1 loss operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SmoothL1LossOperator(Operator):
    """Operator for torch.nn.functional.smooth_l1_loss."""

    def __init__(self):
        super().__init__("nn.functional.smooth_l1_loss")

    def can_produce(self, tensor):
        """smooth_l1_loss can produce scalar or reduced tensors from floating point inputs."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """smooth_l1_loss needs input and target tensors with same shape."""
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
        """Generate code for smooth_l1_loss operation."""
        if len(input_names) != 2:
            raise ValueError("smooth_l1_loss requires exactly 2 inputs (input, target)")

        if len(output_tensor.size) == 0:
            return f"{output_name} = torch.nn.functional.smooth_l1_loss({input_names[0]}, {input_names[1]})"
        else:
            return f"{output_name} = torch.nn.functional.smooth_l1_loss({input_names[0]}, {input_names[1]}, reduction='none')"
