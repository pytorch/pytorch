"""PReLU operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class PreluOperator(Operator):
    """Operator for PReLU activation function (torch.nn.functional.prelu)."""

    def __init__(self):
        super().__init__("prelu")

    def can_produce(self, tensor):
        """PReLU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor and weight tensor for PReLU."""
        # The input to PReLU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # PReLU weight can be either a scalar or have the same number of channels as input
        # For simplicity, let's use a scalar weight (single parameter)
        weight_tensor = Tensor((1,), (1,), tensor.dtype, tensor.device, tensor.supported_ops)

        return [input_tensor, weight_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for PReLU operation."""
        if len(input_names) != 2:
            raise ValueError("PReLU requires 2 inputs (input and weight)")
        return f"{output_name} = torch.nn.functional.prelu({input_names[0]}, {input_names[1]})"
