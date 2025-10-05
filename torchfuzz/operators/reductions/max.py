"""Max operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class MaxOperator(Operator):
    """Operator for tensor max reduction."""

    def __init__(self):
        super().__init__("max")

    def can_produce(self, tensor):
        """
        We construct inputs by inserting at most one extra dimension,
        so we need room to add a dim and stay within a reasonable cap.
        Your generator uses up to 5 dims, so keep input_dim <= 5.
        """
        return len(tensor.size) < 5

    def decompose(self, tensor):
        """
        Construct an input shape that reduces to tensor.size via a max.
        Store the chosen reduction dims on the OUTPUT tensor so codegen can read it.
        """
        if len(tensor.size) == 0:
            # Scalar output, pick an arbitrary input and reduce all dims.
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 5) for _ in range(input_ndim))
            # Mark 'all' to emit .max() with no dim argument.
            tensor._max_dim = "all"
        else:
            # Insert a new dimension of size >= 2 at a random position,
            # then reduce over that single dimension.
            dim = random.randint(0, len(tensor.size))
            expand_size = random.randint(2, 5)
            input_shape = list(tensor.size)
            input_shape.insert(dim, expand_size)
            input_shape = tuple(input_shape)
            tensor._max_dim = dim

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(input_shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for max operation."""
        md = getattr(output_tensor, "_max_dim", None)
        src = input_names[0]
        if md == "all":
            return f"{output_name} = {src}.max()"
        elif isinstance(md, tuple):
            # If you later extend to multi-dim reductions, this handles it.
            return f"{output_name} = {src}.max(dim={md}).values"
        elif isinstance(md, int):
            return f"{output_name} = {src}.max(dim={md}).values"
        else:
            # Safe default for legacy cases: reduce all dims
            return f"{output_name} = {src}.max()"
