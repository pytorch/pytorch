"""SELU operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SeluOperator(Operator):
    """Operator for SELU activation function (torch.nn.functional.selu)."""

    def __init__(self):
        super().__init__("selu")

    def can_produce(self, tensor):
        """SELU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for SELU."""
        # The input to SELU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for SELU operation."""
        return f"{output_name} = torch.nn.functional.selu({input_names[0]})"
