"""Cat operator implementation."""

import random
from ..base.operator import Operator
from torchfuzz.tensor import Tensor


class CatOperator(Operator):
    """Operator for tensor concatenation."""

    def __init__(self):
        super().__init__(supports_dtensor=False)

    def _can_produce_impl(self, output_tensor):
        """Can only cat if there is at least one dimension with size >= 2."""
        return any(s >= 2 for s in output_tensor.size)

    def supports_variable_inputs(self):
        """Cat operator supports variable number of inputs."""
        return True

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for concatenation."""
        # Find all candidate dimensions where size is at least 2
        candidate_dims = [i for i, s in enumerate(tensor.size) if s >= 2]
        if not candidate_dims:
            # No suitable dimension to split, fallback to single tensor
            return [
                Tensor(tensor.size, tensor.stride, tensor.dtype, tensor.device, tensor.supported_ops)
                for _ in range(num_inputs)
            ]

        # Randomly select one of the candidate dimensions
        dim = random.choice(candidate_dims)

        # Randomly choose split points to divide the dimension into num_inputs parts
        total = tensor.size[dim]
        # Ensure each part has at least size 1
        if num_inputs > total:
            num_inputs = total
        # Generate split points
        splits = sorted(random.sample(range(1, total), num_inputs - 1))
        sizes = []
        prev = 0
        for s in splits + [total]:
            sizes.append(s - prev)
            prev = s

        # Build size tuples for each input tensor
        input_sizes = [
            tensor.size[:dim] + (sz,) + tensor.size[dim+1:]
            for sz in sizes
        ]

        # Calculate proper strides for the split tensors
        def calculate_stride(size):
            stride = []
            acc = 1
            for s in reversed(size):
                stride.insert(0, acc)
                acc *= s
            return tuple(stride)

        input_strides = [calculate_stride(sz) for sz in input_sizes]

        tensors = [
            Tensor(sz, st, tensor.dtype, tensor.device, tensor.supported_ops)
            for sz, st in zip(input_sizes, input_strides)
        ]

        # Store the cat dimension on the output tensor instead of input tensors
        tensor._cat_dim = dim
        tensor._cat_sizes = tuple(input_sizes)

        return tensors

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for concatenation operation."""
        # Try to find the dimension along which to cat
        # If the tensor has attribute _cat_dim, use it; else, pick the first valid
        dim = getattr(output_tensor, "_cat_dim", None)
        if dim is None:
            dim = next((i for i, s in enumerate(output_tensor.size) if s >= 2), 0)
        return f"{output_name} = torch.cat([{', '.join(input_names)}], dim={dim})"
