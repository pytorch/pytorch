"""FillDiagonal operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class FillDiagonalOperator_(Operator):
    """Operator for filling tensor diagonal with a value."""

    def __init__(self):
        super().__init__(supports_dtensor=False)

    def _can_produce_impl(self, tensor):
        """
        PyTorch's fill_diagonal_ requires all dimensions to be of equal length.
        Only produce for tensors where all dimensions have the same size.
        """
        if len(tensor.size) < 2:
            return False
        # Check if all dimensions have the same size
        first_size = tensor.size[0]
        return all(s == first_size for s in tensor.size)

    def decompose(self, tensor):
        """Decompose tensor into input tensors for fill_diagonal_ operation."""
        # Find two dimensions with equal size
        dims = None
        for i in range(len(tensor.size)):
            for j in range(i + 1, len(tensor.size)):
                if tensor.size[i] == tensor.size[j]:
                    dims = (i, j)
                    break
            if dims is not None:
                break
        if dims is None:
            # Fallback: just return a tensor of the same shape and a scalar
            return [
                Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops),
                Tensor((), (), tensor.dtype, tensor.device, tensor.supported_ops)
            ]

        # Input 0: tensor to fill (same shape as output)
        # Input 1: value to fill (scalar)
        t_in = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        t_val = Tensor((), (), tensor.dtype, tensor.device, tensor.supported_ops)
        # Store dims for codegen
        tensor._fill_diag_dims = dims
        return [t_in, t_val]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for fill_diagonal_ operation."""
        # PyTorch's fill_diagonal_ expects a scalar number, so use .item() to extract from tensor
        fill_value = f"{input_names[1]}.item()"
        # Since all dimensions are equal now, we can just use fill_diagonal_ directly
        return f"{output_name} = {input_names[0]}.clone(); {output_name}.fill_diagonal_({fill_value})"
