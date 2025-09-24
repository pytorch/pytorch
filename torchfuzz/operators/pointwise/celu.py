"""CELU operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class CeluOperator(Operator):
    """Operator for CELU activation function (torch.nn.functional.celu)."""

    def __init__(self):
        super().__init__("celu")

    def can_produce(self, tensor):
        """CELU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for CELU."""
        # The input to CELU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate random alpha value
        alpha = random.uniform(0.5, 2.0)

        # Store the value for codegen
        self._alpha = alpha

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for CELU operation."""
        alpha = getattr(self, '_alpha', 1.0)  # Default value
        return f"{output_name} = torch.nn.functional.celu({input_names[0]}, alpha={alpha})"
