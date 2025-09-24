"""All operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class AllOperator(Operator):
    """Operator for tensor all reduction."""

    def __init__(self):
        super().__init__("all")

    def can_produce(self, tensor):
        """
        torch.all returns a boolean tensor, so output dtype must be bool.
        We construct inputs by inserting at most one extra dimension,
        so we need room to add a dim and stay within a reasonable cap.
        Your generator uses up to 5 dims, so keep input_dim <= 5.
        """
        return len(tensor.size) < 5 and tensor.dtype == "bool"

    def decompose(self, tensor):
        """
        Construct an input shape that reduces to tensor.size via an all.
        Store the chosen reduction dims on the OUTPUT tensor so codegen can read it.
        """
        if len(tensor.size) == 0:
            # Scalar output, pick an arbitrary input and reduce all dims.
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 5) for _ in range(input_ndim))
            # Mark 'all' to emit .all() with no dim argument.
            tensor._all_dim = "all"
        else:
            # Insert a new dimension of size >= 2 at a random position,
            # then reduce over that single dimension.
            dim = random.randint(0, len(tensor.size))
            expand_size = random.randint(2, 5)
            input_shape = list(tensor.size)
            input_shape.insert(dim, expand_size)
            input_shape = tuple(input_shape)
            tensor._all_dim = dim

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(input_shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        # Input tensor should be bool for torch.all
        t_in = Tensor(input_shape, stride, "bool", tensor.device, tensor.supported_ops)
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for all operation."""
        ad = getattr(output_tensor, "_all_dim", None)
        src = input_names[0]
        if ad == "all":
            return f"{output_name} = {src}.all()"
        elif isinstance(ad, tuple):
            # If you later extend to multi-dim reductions, this handles it.
            return f"{output_name} = {src}.all(dim={ad})"
        elif isinstance(ad, int):
            return f"{output_name} = {src}.all(dim={ad})"
        else:
            # Safe default for legacy cases: reduce all dims
            return f"{output_name} = {src}.all()"
