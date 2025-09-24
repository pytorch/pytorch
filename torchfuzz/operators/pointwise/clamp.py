"""Clamp operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ClampOperator(Operator):
    """Operator for clamping tensor values (torch.clamp)."""

    def __init__(self):
        super().__init__("clamp")

    def can_produce(self, tensor):
        """Clamp can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for clamp."""
        # The input to clamp must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # Generate random min and max values for clamping
        if tensor.dtype in ["float16", "float32", "float64", "bfloat16"]:
            # For floating point types, use reasonable float ranges
            min_val = random.uniform(-10.0, 0.0)
            max_val = random.uniform(0.0, 10.0)
        else:
            # For integer types, use integer ranges
            min_val = random.randint(-100, 0)
            max_val = random.randint(1, 100)

        # Store the values for codegen
        self._min = min_val
        self._max = max_val

        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for clamp operation."""
        min_val = getattr(self, '_min', None)
        max_val = getattr(self, '_max', None)

        if min_val is not None and max_val is not None:
            return f"{output_name} = torch.clamp({input_names[0]}, min={min_val}, max={max_val})"
        elif min_val is not None:
            return f"{output_name} = torch.clamp({input_names[0]}, min={min_val})"
        elif max_val is not None:
            return f"{output_name} = torch.clamp({input_names[0]}, max={max_val})"
        else:
            # Fallback with default values
            return f"{output_name} = torch.clamp({input_names[0]}, min=-1.0, max=1.0)"
