"""Mish operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class MishOperator(Operator):
    """Operator for Mish activation function (torch.nn.functional.mish)."""

    def __init__(self):
        super().__init__("mish")

    def can_produce(self, tensor):
        """Mish can be applied only to float tensors (elementwise op)."""
        # Only allow float dtypes (e.g., float32, float64)
        return tensor.dtype in ("float32")

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Mish."""
        # The input to Mish must have the same shape, dtype, and device as the output
        # Ensure the dtype is float32, since Mish is not implemented for int64/Long
        input_tensor = Tensor(
            tensor.size,
            tensor.stride,
            "float32",  # force float32 dtype for Mish compatibility
            tensor.device,
            tensor.supported_ops,
        )
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Mish operation."""
        return f"{output_name} = torch.nn.functional.mish({input_names[0]})"
