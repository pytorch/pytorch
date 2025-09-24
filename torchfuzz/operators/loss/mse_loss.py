"""Mean squared error loss operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class MseLossOperator(Operator):
    """Operator for torch.nn.functional.mse_loss."""

    def __init__(self):
        super().__init__("nn.functional.mse_loss")

    def can_produce(self, tensor):
        """mse_loss can produce scalar or reduced tensors from floating point inputs."""
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """mse_loss needs input and target tensors with same shape."""
        # For scalar output, input and target can be any shape
        # For non-scalar output, input and target have same shape as output
        if len(tensor.size) == 0:
            # Scalar output - input/target can be any shape
            input_size = (4, 4)  # Arbitrary shape
        else:
            # Non-scalar output - input/target have same shape as output
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
        """Generate code for mse_loss operation."""
        if len(input_names) != 2:
            raise ValueError("mse_loss requires exactly 2 inputs (input, target)")

        if len(output_tensor.size) == 0:
            # Scalar output - use default reduction
            return f"{output_name} = torch.nn.functional.mse_loss({input_names[0]}, {input_names[1]})"
        else:
            # Non-scalar output - use reduction='none'
            return f"{output_name} = torch.nn.functional.mse_loss({input_names[0]}, {input_names[1]}, reduction='none')"
