"""View operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ViewOperator(Operator):
    """Operator for tensor view/reshape operations."""

    def __init__(self):
        super().__init__("view")

    def can_produce(self, tensor):
        """View can always target any shape with the same numel."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for view operation."""
        # Pick an input shape with the same numel as tensor.size
        numel = 1
        for s in tensor.size:
            numel *= s

        # Try to factor numel into 1-3 dims
        for _ in range(10):
            ndims = random.randint(1, 3)

            def random_shape(n, d):
                if d == 1:
                    return (n,)
                factors = []
                rem = n
                for i in range(d - 1):
                    divisors = [f for f in range(1, rem + 1) if rem % f == 0]
                    f = random.choice(divisors)
                    factors.append(f)
                    rem //= f
                factors.append(rem)
                return tuple(factors)

            shape = random_shape(numel, ndims)
            if all(isinstance(x, int) and x > 0 for x in shape):
                break
        else:
            shape = (numel,)

        # contiguous stride for input
        stride = []
        acc = 1
        for s in reversed(shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)

        t_in = Tensor(shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
        # No metadata needed on inputs. Codegen will view to the output size.
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for view operation."""
        # Always view to the output's shape
        return f"{output_name} = {input_names[0]}.view({tuple(output_tensor.size)})"
