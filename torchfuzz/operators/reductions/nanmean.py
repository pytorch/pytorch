"""Nanmean operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class NanmeanOperator(Operator):
    """Operator for tensor nanmean reduction."""

    def __init__(self):
        super().__init__("nanmean")

    def can_produce(self, tensor):
        """
        We construct inputs by inserting at most one extra dimension,
        so we need room to add a dim and stay within a reasonable cap.
        Your generator uses up to 5 dims, so keep input_dim <= 5.
        Also, nanmean only supports floating point and complex dtypes.
        """
        # Check dtype compatibility
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) < 5

    def decompose(self, tensor):
        """
        Construct an input shape that reduces to tensor.size via a nanmean.
        Store the chosen reduction dims on the OUTPUT tensor so codegen can read it.
        """
        if len(tensor.size) == 0:
            # Scalar output, pick an arbitrary input and reduce all dims.
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 5) for _ in range(input_ndim))
            # Mark 'all' to emit .nanmean() with no dim argument.
            tensor._nanmean_dim = "all"
        else:
            # Insert a new dimension of size >= 2 at a random position,
            # then reduce over that single dimension.
            dim = random.randint(0, len(tensor.size))
            expand_size = random.randint(2, 5)
            input_shape = list(tensor.size)
            input_shape.insert(dim, expand_size)
            input_shape = tuple(input_shape)
            tensor._nanmean_dim = dim

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
        """Generate code for nanmean operation."""
        md = getattr(output_tensor, "_nanmean_dim", None)
        src = input_names[0]
        if md == "all":
            return f"{output_name} = {src}.nanmean()"
        elif isinstance(md, tuple):
            # If you later extend to multi-dim reductions, this handles it.
            return f"{output_name} = {src}.nanmean(dim={md})"
        elif isinstance(md, int):
            return f"{output_name} = {src}.nanmean(dim={md})"
        else:
            # Safe default for legacy cases: reduce all dims
            return f"{output_name} = {src}.nanmean()"
