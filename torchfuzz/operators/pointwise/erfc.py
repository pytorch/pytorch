"""Erfc operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ErfcOperator(Operator):
    """Operator for complementary error function."""

    def __init__(self):
        super().__init__("erfc")

    def can_produce(self, tensor):
        """Erfc can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Erfc."""
        # The input to Erfc must have the same shape, stride, dtype, and device as the output
        return [
            Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        ]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Erfc operation."""
        # Use torch.erfc for the complementary error function operation
        return f"{output_name} = torch.erfc({input_names[0]})"
