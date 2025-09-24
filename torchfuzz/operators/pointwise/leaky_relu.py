"""Leaky ReLU operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class LeakyReluOperator(Operator):
    """Operator for Leaky ReLU activation function (torch.nn.functional.leaky_relu)."""

    def __init__(self):
        super().__init__("leaky_relu")

    def can_produce(self, tensor):
        """Leaky ReLU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Leaky ReLU."""
        # The input to Leaky ReLU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate random negative slope value
        negative_slope = random.uniform(0.01, 0.3)

        # Store the value for codegen
        self._negative_slope = negative_slope

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Leaky ReLU operation."""
        negative_slope = getattr(self, '_negative_slope', 0.01)  # Default value
        return f"{output_name} = torch.nn.functional.leaky_relu({input_names[0]}, negative_slope={negative_slope})"
