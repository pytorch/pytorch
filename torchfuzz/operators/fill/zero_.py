"""Zero operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class ZeroOperator_(Operator):
    """Operator for filling tensor with zeros."""

    def __init__(self):
        super().__init__("zero_")

    def can_produce(self, tensor):
        """Zero can always produce a tensor by filling with zeros."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensors for zero_ operation."""
        # Input 0: tensor to fill (same shape as output)
        t_in = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for zero_ operation."""
        return f"{output_name} = {input_names[0]}.clone(); {output_name}.zero_()"
