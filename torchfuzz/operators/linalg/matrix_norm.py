"""Linear algebra matrix norm operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MatrixNormOperator(Operator):
    """Operator for torch.linalg.matrix_norm."""

    def __init__(self):
        super().__init__("linalg.matrix_norm")

    def can_produce(self, tensor):
        """linalg.matrix_norm can produce scalars or reduced tensors from floating point matrix inputs."""
        # linalg.matrix_norm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Can produce any dimensionality output depending on dim parameter
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for matrix norm calculation."""
        # The input must be at least 2D for matrix norm
        if len(tensor.size) == 0:
            # Scalar output - input is a 2D matrix
            input_size = (random.randint(2, 8), random.randint(2, 8))
        else:
            # Non-scalar output - add matrix dimensions to existing batch dimensions
            input_size = tensor.size + (random.randint(2, 8), random.randint(2, 8))

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
        """Generate code for linalg.matrix_norm operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.matrix_norm requires exactly 1 input")

        # Generate appropriate dim parameter based on output shape
        if len(output_tensor.size) == 0:
            # Scalar output - compute norm over matrix dimensions
            return f"{output_name} = torch.linalg.matrix_norm({input_names[0]})"
        else:
            # Non-scalar output - compute norm over last two dimensions, keep batch dimensions
            return f"{output_name} = torch.linalg.matrix_norm({input_names[0]}, dim=(-2, -1))"
