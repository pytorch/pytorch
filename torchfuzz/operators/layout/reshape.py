"""Reshape operator implementation."""

import random
from ..base import Operator
from torchfuzz.tensor import Tensor


class ReshapeOperator(Operator):
    """Operator for tensor reshape operations."""

    def __init__(self):
        super().__init__("reshape")

    def can_produce(self, tensor):
        """Reshape can always target any shape with the same numel."""
        return True

    def decompose(self, tensor):
        """Decompose tensor into input tensor for reshape operation."""
        # Pick an input shape with the same numel as tensor.size
        numel = 1
        for s in tensor.size:
            numel *= s

        # Handle special case where numel is 0
        if numel == 0:
            # For zero-size tensors, create a simple input shape with one zero dimension
            ndims = random.randint(1, 4)
            if ndims == 1:
                shape = (0,)
            else:
                # Create a shape with one zero dimension and others as 1
                zero_pos = random.randint(0, ndims - 1)
                shape = tuple(1 if i != zero_pos else 0 for i in range(ndims))
        else:
            # Try to factor numel into 1-4 dims
            for _ in range(10):
                ndims = random.randint(1, 4)

                def random_shape(n, d):
                    if d == 1:
                        return (n,)
                    if n <= 0:
                        return (n,) if n == 0 else (1,)

                    factors = []
                    rem = n
                    for i in range(d - 1):
                        if rem <= 0:
                            break
                        divisors = [f for f in range(1, rem + 1) if rem % f == 0]
                        if not divisors:  # Safety check for empty divisors
                            return (n,)  # Fallback to 1D
                        f = random.choice(divisors)
                        factors.append(f)
                        rem //= f

                    if rem > 0:
                        factors.append(rem)

                    # Pad with 1s if we don't have enough dimensions
                    while len(factors) < d:
                        factors.append(1)

                    return tuple(factors[:d])  # Ensure we don't exceed d dimensions

                shape = random_shape(numel, ndims)
                if all(isinstance(x, int) and x >= 0 for x in shape):
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
        # No metadata needed on inputs. Codegen will reshape to the output size.
        return [t_in]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for reshape operation."""
        # Always reshape to the output's shape
        return f"{output_name} = {input_names[0]}.reshape({tuple(output_tensor.size)})"
