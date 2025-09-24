"""Argmax operator implementation."""

import random
from ..base.operator import Operator
from torchfuzz.tensor import Tensor


class ArgmaxOperator(Operator):
    """Operator for tensor argmax reduction."""

    def __init__(self):
        super().__init__("argmax", supports_dtensor=True)

    def _can_produce_impl(self, tensor):
        """
        Argmax returns indices, so output dtype must be int64.
        We construct inputs by inserting at most one extra dimension,
        so we need room to add a dim and stay within a reasonable cap.
        Your generator uses up to 5 dims, so keep input_dim <= 5.
        """
        return len(tensor.size) < 5 and tensor.dtype == "int64"

    def decompose(self, tensor):
        """
        Construct an input shape that reduces to tensor.size via an argmax.
        Store the chosen reduction dims on the OUTPUT tensor so codegen can read it.
        """
        if len(tensor.size) == 0:
            # Scalar output, pick an arbitrary input and reduce all dims.
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 5) for _ in range(input_ndim))
            # Mark 'all' to emit .argmax() with no dim argument.
            tensor._argmax_dim = "all"
        else:
            # Insert a new dimension of size >= 2 at a random position,
            # then reduce over that single dimension.
            dim = random.randint(0, len(tensor.size))
            expand_size = random.randint(2, 5)
            input_shape = list(tensor.size)
            input_shape.insert(dim, expand_size)
            input_shape = tuple(input_shape)
            tensor._argmax_dim = dim

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        # Input tensor can be float32, bfloat16, or float16 (common types for argmax)
        input_dtypes = ["float32", "bfloat16", "float16"]
        input_dtype = random.choice(input_dtypes)

        t_in = Tensor(input_shape, stride, input_dtype, tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for argmax operation."""
        ad = getattr(output_tensor, "_argmax_dim", None)
        src = input_names[0]
        if ad == "all":
            return f"{output_name} = {src}.argmax()"
        elif isinstance(ad, tuple):
            # If you later extend to multi-dim reductions, this handles it.
            return f"{output_name} = {src}.argmax(dim={ad})"
        elif isinstance(ad, int):
            return f"{output_name} = {src}.argmax(dim={ad})"
        else:
            # Safe default for legacy cases: reduce all dims
            return f"{output_name} = {src}.argmax()"
