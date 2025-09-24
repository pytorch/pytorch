"""ELU operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class EluOperator(Operator):
    """Operator for ELU activation function (torch.nn.functional.elu)."""

    def __init__(self):
        super().__init__("elu")

    def can_produce(self, tensor):
        """ELU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for ELU."""
        # The input to ELU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate random alpha value
        alpha = random.uniform(0.5, 2.0)

        # Store the value for codegen
        self._alpha = alpha

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for ELU operation."""
        alpha = getattr(self, '_alpha', 1.0)  # Default value
        return f"{output_name} = torch.nn.functional.elu({input_names[0]}, alpha={alpha})"
