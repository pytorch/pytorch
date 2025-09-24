"""Random normal operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RandnOperator(Operator):
    """Operator for torch.randn."""

    def __init__(self):
        super().__init__("randn")

    def can_produce(self, tensor):
        """torch.randn can produce any floating point tensor."""
        # torch.randn only produces floating point tensors
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.randn generates tensors without input, return empty list."""
        # torch.randn is a generator function, no input tensors needed
        return []

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.randn operation."""
        if len(input_names) != 0:
            raise ValueError("torch.randn requires no inputs")

        # Generate the size tuple
        size_str = str(output_tensor.size)
        dtype_str = f"dtype=torch.{output_tensor.dtype}"
        device_str = f"device='{output_tensor.device}'"

        return f"{output_name} = torch.randn({size_str}, {dtype_str}, {device_str})"
