"""Instance normalization operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class InstanceNormOperator(Operator):
    """Operator for instance normalization (torch.nn.functional.instance_norm)."""

    def __init__(self):
        super().__init__("instance_norm")

    def can_produce(self, tensor):
        """InstanceNorm can produce tensors that are at least 3D (N, C, ...) with floating point types."""
        # Instance norm only supports floating point dtypes
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Instance norm requires at least 3D tensors (N, C, ...)
        return len(tensor.size) >= 3

    def decompose(self, tensor):
        """Decompose tensor into input tensor for instance norm operation."""
        # The input to instance norm must have the same shape, dtype, and device as the output
        input_tensor = Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for instance norm operation."""
        return f"{output_name} = torch.nn.functional.instance_norm({input_names[0]})"
