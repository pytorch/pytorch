"""Cross entropy loss operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class CrossEntropyOperator(Operator):
    """Operator for torch.nn.functional.cross_entropy."""

    def __init__(self):
        super().__init__("nn.functional.cross_entropy")

    def can_produce(self, tensor):
        """cross_entropy can produce scalar or reduced tensors from floating point inputs."""
        # cross_entropy produces floating point tensors
        return tensor.dtype in ["float16", "float32", "float64", "bfloat16"]

    def decompose(self, tensor):
        """cross_entropy needs input logits and target class indices."""
        # Input shape: (N, C) for 1D case, (N, C, ...) for higher dimensions
        # Target shape: (N,) for 1D case, (N, ...) for higher dimensions
        # Output shape: () for scalar, (N, ...) for unreduced

        if len(tensor.size) == 0:
            # Scalar output (default reduction='mean' or 'sum')
            N = random.randint(2, 8)
            C = random.randint(2, 10)
            input_size = (N, C)
            target_size = (N,)
        else:
            # Non-scalar output (reduction='none')
            N = tensor.size[0] if len(tensor.size) > 0 else random.randint(2, 8)
            C = random.randint(2, 10)
            batch_dims = tensor.size
            input_size = (N, C) + batch_dims[1:] if len(batch_dims) > 1 else (N, C)
            target_size = batch_dims

        # Calculate strides for contiguous tensors
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)
        target_stride = calc_stride(target_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        target_tensor = Tensor(target_size, target_stride, "int64", tensor.device, tensor.supported_ops)

        return [input_tensor, target_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for cross_entropy operation."""
        if len(input_names) != 2:
            raise ValueError("cross_entropy requires exactly 2 inputs (input, target)")

        if len(output_tensor.size) == 0:
            # Scalar output - use default reduction
            return f"{output_name} = torch.nn.functional.cross_entropy({input_names[0]}, {input_names[1]})"
        else:
            # Non-scalar output - use reduction='none'
            return f"{output_name} = torch.nn.functional.cross_entropy({input_names[0]}, {input_names[1]}, reduction='none')"
