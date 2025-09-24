"""SiLU (Swish) operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class SiluOperator(Operator):
    """Operator for SiLU (Swish) activation function (torch.nn.functional.silu)."""

    def __init__(self):
        super().__init__("silu")

    def can_produce(self, tensor):
        """SiLU can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for SiLU."""
        # The input to SiLU must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for SiLU operation."""
        return f"{output_name} = torch.nn.functional.silu({input_names[0]})"
