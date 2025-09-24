"""Linear algebra norm operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class NormOperator(Operator):
    """Operator for torch.linalg.norm."""

    def __init__(self):
        super().__init__("linalg.norm")

    def can_produce(self, tensor):
        """linalg.norm can produce scalars or reduced tensors from floating point inputs."""
        # linalg.norm only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        # Can produce any dimensionality output depending on dim parameter
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for norm calculation."""
        # The input can have additional dimensions that get reduced
        # We'll create an input that's larger than the output
        if len(tensor.size) == 0:
            # Scalar output - input can be any shape
            input_size = random.choice([
                (random.randint(2, 8),),
                (random.randint(2, 8), random.randint(2, 8)),
                (random.randint(2, 8), random.randint(2, 8), random.randint(2, 8))
            ])
        else:
            # Non-scalar output - add one or more dimensions
            extra_dims = random.randint(1, 2)
            input_size = tuple([random.randint(2, 8) for _ in range(extra_dims)]) + tensor.size

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
        """Generate code for linalg.norm operation."""
        if len(input_names) != 1:
            raise ValueError("linalg.norm requires exactly 1 input")

        # Generate appropriate dim parameter based on output shape
        input_tensor = self.decompose(output_tensor)[0]  # Get the input tensor we would create

        if len(output_tensor.size) == 0:
            # Scalar output - compute norm over all dimensions
            return f"{output_name} = torch.linalg.norm({input_names[0]})"
        else:
            # Non-scalar output - compute norm over some dimensions
            input_ndim = len(input_tensor.size)
            output_ndim = len(output_tensor.size)
            dims_to_reduce = input_ndim - output_ndim

            if dims_to_reduce == 1:
                dim = 0  # Reduce first dimension
                return f"{output_name} = torch.linalg.norm({input_names[0]}, dim={dim})"
            else:
                dims = list(range(dims_to_reduce))
                return f"{output_name} = torch.linalg.norm({input_names[0]}, dim={dims})"
