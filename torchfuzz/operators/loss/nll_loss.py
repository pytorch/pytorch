"""Negative log likelihood loss operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class NllLossOperator(Operator):
    """Operator for torch.nn.functional.nll_loss."""

    def __init__(self):
        super().__init__("nn.functional.nll_loss")

    def can_produce(self, tensor):
        """nll_loss can produce scalar or reduced tensors from floating point inputs."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """nll_loss needs input and target tensors with compatible shapes."""
        if len(tensor.size) == 0:
            # Scalar output - input can be (N, C) and target (N,)
            input_size = (4, 5)  # (batch_size, num_classes)
            target_size = (4,)   # (batch_size,)
        else:
            # Non-scalar output - input and target have compatible shapes
            input_size = tensor.size + (5,)  # Add class dimension
            target_size = tensor.size

        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        target_stride = calc_stride(target_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        target_tensor = Tensor(target_size, target_stride, "int64", tensor.device, tensor.supported_ops)

        return [input_tensor, target_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for nll_loss operation."""
        if len(input_names) != 2:
            raise ValueError("nll_loss requires exactly 2 inputs (input, target)")

        if len(output_tensor.size) == 0:
            return f"{output_name} = torch.nn.functional.nll_loss({input_names[0]}, {input_names[1]})"
        else:
            return f"{output_name} = torch.nn.functional.nll_loss({input_names[0]}, {input_names[1]}, reduction='none')"
