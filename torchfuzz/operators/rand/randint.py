"""Random integer operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RandintOperator(Operator):
    """Operator for torch.randint."""

    def __init__(self):
        super().__init__("randint")

    def can_produce(self, tensor):
        """torch.randint can produce any integer tensor."""
        # torch.randint produces integer tensors
        return tensor.dtype in ["int8", "int16", "int32", "int64", "uint8"]

    def decompose(self, tensor):
        """torch.randint generates tensors without input, return empty list."""
        # torch.randint is a generator function, no input tensors needed
        return []

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.randint operation."""
        if len(input_names) != 0:
            raise ValueError("torch.randint requires no inputs")

        # Generate the size tuple and parameters
        size_str = str(output_tensor.size)
        dtype_str = f"dtype=torch.{output_tensor.dtype}"
        device_str = f"device='{output_tensor.device}'"

        # Use default range of 0 to 10 for simplicity
        return f"{output_name} = torch.randint(0, 10, {size_str}, {dtype_str}, {device_str})"
