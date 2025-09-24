"""Random uniform operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class RandOperator(Operator):
    """Operator for torch.rand."""

    def __init__(self):
        super().__init__("rand")

    def can_produce(self, tensor):
        """torch.rand can produce any floating point tensor."""
        # torch.rand only produces floating point tensors
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """torch.rand generates tensors without input, return empty list."""
        # torch.rand is a generator function, no input tensors needed
        return []

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for torch.rand operation."""
        if len(input_names) != 0:
            raise ValueError("torch.rand requires no inputs")

        # Generate the size tuple
        size_str = str(output_tensor.size)
        dtype_str = f"dtype=torch.{output_tensor.dtype}"
        device_str = f"device='{output_tensor.device}'"

        return f"{output_name} = torch.rand({size_str}, {dtype_str}, {device_str})"
