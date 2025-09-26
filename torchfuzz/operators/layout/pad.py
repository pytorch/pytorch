"""Pad operator implementation."""

from ..base import Operator
from torchfuzz.tensor import Tensor


class PadOperator(Operator):
    """Operator for padding (torch.nn.functional.pad)."""

    def __init__(self):
        super().__init__("pad")

    def can_produce(self, tensor):
        """Pad can be applied to any tensor with at least 1 dimension."""
        # Only produce tensors where we can apply minimal single-dimension padding
        # This means we can only reliably produce outputs where last dimension > 1
        return len(tensor.size) >= 1 and tensor.size[-1] > 1

    def decompose(self, tensor):
        """Decompose tensor into input tensor for pad operation."""
        # Simple approach: create an input tensor with same size as output
        # Apply minimal padding to ensure the operation is valid
        
        # For now, just apply minimal padding to the last dimension to avoid complex verification issues
        input_size = list(tensor.size)
        
        # Apply small padding to the last dimension only
        if tensor.size[-1] > 1:
            pad_values = [0, 1]  # Add 1 to the right of last dimension
            input_size[-1] = tensor.size[-1] - 1  # Input is 1 smaller
        else:
            pad_values = [1, 0]  # Add 1 to the left of last dimension
            # Keep same input size - output will be 1 larger than requested
        
        input_stride = self._calc_stride(tuple(input_size))
        input_tensor = Tensor(tuple(input_size), input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        
        # Store the padding values so codegen can use them
        self._pad_values = pad_values
        
        return [input_tensor]

    def _calc_stride(self, size):
        """Calculate stride for contiguous tensor."""
        stride = [1]
        for dim in reversed(size[:-1]):
            stride.insert(0, stride[0] * dim)
        return tuple(stride)

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for pad operation."""
        mode = 'constant'
        value = 0.0

        # Use the padding values calculated in decompose
        pad = getattr(self, '_pad_values', [0, 1])

        return f"{output_name} = torch.nn.functional.pad({input_names[0]}, {pad}, mode='{mode}', value={value})"
