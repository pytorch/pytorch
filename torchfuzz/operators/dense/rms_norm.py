"""RMSNorm operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class RmsNormOperator(Operator):
    """Operator for RMS normalization (torch.rms_norm)."""

    def __init__(self):
        super().__init__("rms_norm")

    def can_produce(self, tensor):
        """RMSNorm supports floating and complex types; require at least 2 dims."""
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) >= 2

    def decompose(self, tensor):
        """Decompose tensor into input and normalized_shape for rms_norm."""
        input_size = tensor.size
        ndim = len(input_size)
        if ndim >= 3:
            norm_dims = random.choice([1, 2])
        else:
            norm_dims = 1
        normalized_shape = tuple(input_size[-norm_dims:])

        # contiguous stride for input
        input_stride = self._calc_stride(input_size)
        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)

        # store for codegen
        self._normalized_shape = normalized_shape
        return [input_tensor]

    def _calc_stride(self, size):
        stride = [1]
        for dim in reversed(size[:-1]):
            stride.insert(0, stride[0] * dim)
        return tuple(stride)

    def codegen(self, output_name, input_names, output_tensor):
        normalized_shape = getattr(self, "_normalized_shape", output_tensor.size[-1:])
        # validate shape matches tail dims; otherwise fallback to last dim
        input_shape = output_tensor.size
        if len(normalized_shape) > len(input_shape) or normalized_shape != tuple(input_shape[-len(normalized_shape):]):
            normalized_shape = (input_shape[-1],)
        return f"{output_name} = torch.rms_norm({input_names[0]}, {normalized_shape})"
