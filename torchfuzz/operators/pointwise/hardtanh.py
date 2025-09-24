"""Hardtanh operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class HardtanhOperator(Operator):
    """Operator for Hardtanh activation function (torch.nn.functional.hardtanh)."""

    def __init__(self):
        super().__init__("hardtanh")

    def can_produce(self, tensor):
        """Hardtanh can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Hardtanh."""
        # The input to Hardtanh must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate random min and max values
        min_val = random.uniform(-2.0, -0.5)
        max_val = random.uniform(0.5, 2.0)

        # Store the values for codegen
        self._min_val = min_val
        self._max_val = max_val

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Hardtanh operation."""
        min_val = getattr(self, '_min_val', -1.0)  # Default values
        max_val = getattr(self, '_max_val', 1.0)
        return f"{output_name} = torch.nn.functional.hardtanh({input_names[0]}, min_val={min_val}, max_val={max_val})"
