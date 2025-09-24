"""GELU operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class GeluOperator(Operator):
    """Operator for GELU activation function. DTensor-safe operation."""

    def __init__(self):
        super().__init__(supports_dtensor=True)

    def _can_produce_impl(self, output_tensor):
        """GELU can be applied to floating point tensors (elementwise op)."""
        # GELU only supports floating point and complex dtypes
        if output_tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for GELU."""
        # The input to GELU must have the same shape, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for GELU operation."""
        # Use torch.nn.functional.gelu for the GELU activation
        return f"{output_name} = torch.nn.functional.gelu({input_names[0]})"
