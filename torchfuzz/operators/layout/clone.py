"""Clone operator implementation."""

from ..base import Operator
from tensor import Tensor


class CloneOperator(Operator):
    """Operator for tensor clone operations."""

    def __init__(self):
        super().__init__("clone")

    def can_produce(self, tensor):
        """Clone can produce any tensor (creates an exact copy)."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for clone operation."""
        # Clone produces an exact copy, so input tensor should be identical
        # in shape, dtype, device, and strides
        t_in = Tensor(
            tensor.size,
            tensor.stride,
            tensor.dtype,
            tensor.device,
            tensor.supported_ops
        )
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for clone operation."""
        return f"{output_name} = {input_names[0]}.clone()"
