"""Linear algebra vector norm operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class VectorNormOperator(Operator):
    """Operator for torch.linalg.vector_norm."""

    def __init__(self):
        super().__init__("linalg.vector_norm")

    def can_produce(self, tensor):
        """linalg.vector_norm can produce scalars or reduced tensors from floating point inputs."""
        # linalg.vector_norm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Can produce any dimensionality output depending on dim parameter
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for vector norm calculation."""
        # The input can have additional dimensions that get reduced
        if len(tensor.size) == 0:
            # Scalar output - input can be any shape
            input_size = random.choice([
                (random.randint(2, 8),),
                (random.randint(2, 8), random.randint(2, 8)),
                (random.randint(2, 8), random.randint(2, 8), random.randint(2, 8))
            ])
        else:
            # Non-scalar output - add one dimension
            input_size = (random.randint(2, 8),) + tensor.size

        # Calculate stride for contiguous tensor
        def calc_stride(size):
            stride = [1]
            for dim in reversed(size[:-1]):
                stride.insert(0, stride[0] * dim)
            return tuple(stride)

        input_stride = calc_stride(input_size)

        input_tensor = Tensor(input_size, input_stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [input_tensor]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for linalg.vector_norm operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.vector_norm requires exactly 1 input")

        # Generate appropriate dim parameter based on output shape
        if len(output_tensor.size) == 0:
            # Scalar output - compute norm over all dimensions
            return f"{output_name} = torch.linalg.vector_norm({input_names[0]})"
        else:
            # Non-scalar output - compute norm over first dimension
            return f"{output_name} = torch.linalg.vector_norm({input_names[0]}, dim=0)"
