"""Fill operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class FillOperator_(Operator):
    """Operator for filling tensor with a scalar value."""

    def __init__(self):
        super().__init__(supports_dtensor=False)

    def _can_produce_impl(self, tensor):
        """Fill can always produce a tensor by filling with a scalar value."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensors for fill_ operation."""
        # Input 0: tensor to fill (same shape as output)
        # Input 1: scalar value to fill with
        t_in = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        t_val = Tensor((), (), tensor.dtype, tensor.device, tensor.supported_ops)
        return [t_in, t_val]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for fill_ operation."""
        # PyTorch's fill_ expects a scalar number, so use .item() to extract from tensor
        fill_value = f"{input_names[1]}.item()"
        return f"{output_name} = {input_names[0]}.clone(); {output_name}.fill_({fill_value})"
