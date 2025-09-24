"""Hardsigmoid operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class HardsigmoidOperator(Operator):
    """Operator for Hardsigmoid activation function (torch.nn.functional.hardsigmoid)."""

    def __init__(self):
        super().__init__("hardsigmoid")

    def can_produce(self, tensor):
        """Hardsigmoid can be applied to any tensor (elementwise op)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for Hardsigmoid."""
        # The input to Hardsigmoid must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for Hardsigmoid operation."""
        return f"{output_name} = torch.nn.functional.hardsigmoid({input_names[0]})"
