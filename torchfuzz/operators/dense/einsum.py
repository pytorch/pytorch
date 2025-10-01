"""Einsum operator implementation."""

import random
import string
from ..base import Operator
from tensor import Tensor


class EinsumOperator(Operator):
    """Operator for Einstein summation (torch.einsum)."""

    def __init__(self):
        super().__init__("einsum")

    def can_produce(self, tensor):
        """Einsum can produce tensors up to 4D with floating point types."""
        # einsum only supports floating point tensors
        if tensor.dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
            return False
        return len(tensor.size) <= 4

    def decompose(self, tensor, num_inputs=2):
        """Decompose tensor into input tensors for einsum operation."""
        if num_inputs < 1 or num_inputs > 3:
            # Limit to 1-3 inputs for practical complexity
            num_inputs = random.randint(1, 3)

        output_shape = tensor.size

        # Generate einsum equation based on common patterns
        if num_inputs == 1:
            # Single input operations: transpose, trace, diagonal, etc.
            equation, input_shapes = self._generate_single_input_equation(output_shape)
        elif num_inputs == 2:
            # Two input operations: matrix multiply, batch multiply, outer product, etc.
            equation, input_shapes = self._generate_two_input_equation(output_shape)
        else:  # num_inputs == 3
            # Three input operations: generalized tensor contractions
            equation, input_shapes = self._generate_three_input_equation(output_shape)

        # Store the equation on the output tensor for codegen
        tensor._einsum_equation = equation

        # Create input tensors
        input_tensors = []
        for shape in input_shapes:
            # Calculate contiguous stride
            stride = []
            acc = 1
            for s in reversed(shape):
                stride.insert(0, acc)
                acc *= s
            stride = tuple(stride)

            input_tensor = Tensor(shape, stride, tensor.dtype, tensor.device, tensor.supported_ops)
            input_tensors.append(input_tensor)

        return input_tensors

    def _generate_single_input_equation(self, output_shape):
        """Generate einsum equation for single input operations."""
        output_ndim = len(output_shape)

        if output_ndim == 0:
            # Scalar output - trace or sum all dimensions
            input_ndim = random.randint(1, 3)
            input_shape = tuple(random.randint(2, 8) for _ in range(input_ndim))

            # Generate equation that sums all dimensions
            indices = list(string.ascii_lowercase[:input_ndim])
            equation = f"{''.join(indices)}->"
            return equation, [input_shape]

        elif output_ndim == 1:
            # Vector output - could be diagonal extraction or dimension reduction
            if random.random() < 0.5:
                # Diagonal extraction: "ii->i"
                dim_size = output_shape[0]
                input_shape = (dim_size, dim_size)
                equation = "ii->i"
                return equation, [input_shape]
            else:
                # Dimension reduction: "ij->i" or "ijk->i"
                dim_size = output_shape[0]
                extra_dims = random.randint(1, 2)
                input_shape = [dim_size] + [random.randint(2, 8) for _ in range(extra_dims)]

                indices = list(string.ascii_lowercase[:len(input_shape)])
                equation = f"{''.join(indices)}->{indices[0]}"
                return equation, [tuple(input_shape)]

        elif output_ndim == 2:
            # Matrix output - transpose or partial reduction
            if random.random() < 0.5:
                # Transpose: "ji->ij"
                equation = "ji->ij"
                input_shape = (output_shape[1], output_shape[0])
                return equation, [input_shape]
            else:
                # Partial reduction: "ijk->ij"
                extra_dim = random.randint(2, 8)
                input_shape = output_shape + (extra_dim,)
                equation = "ijk->ij"
                return equation, [input_shape]

        else:  # output_ndim >= 3
            # Multi-dimensional tensor - partial reduction
            extra_dim = random.randint(2, 8)
            input_shape = output_shape + (extra_dim,)
            indices = list(string.ascii_lowercase[:len(input_shape)])
            output_indices = indices[:-1]  # Remove last dimension
            equation = f"{''.join(indices)}->{''.join(output_indices)}"
            return equation, [input_shape]

    def _generate_two_input_equation(self, output_shape):
        """Generate einsum equation for two input operations."""
        output_ndim = len(output_shape)

        if output_ndim == 0:
            # Scalar output - inner product
            vec_size = random.randint(2, 64)
            input_shape1 = (vec_size,)
            input_shape2 = (vec_size,)
            equation = "i,i->"
            return equation, [input_shape1, input_shape2]

        elif output_ndim == 1:
            # Vector output - matrix-vector multiply
            vec_size = output_shape[0]
            inner_dim = random.randint(2, 64)
            input_shape1 = (vec_size, inner_dim)
            input_shape2 = (inner_dim,)
            equation = "ij,j->i"
            return equation, [input_shape1, input_shape2]

        elif output_ndim == 2:
            # Matrix output - matrix multiply or outer product
            if random.random() < 0.8:
                # Matrix multiply: "ik,kj->ij"
                m, n = output_shape
                k = random.randint(2, 64)
                input_shape1 = (m, k)
                input_shape2 = (k, n)
                equation = "ik,kj->ij"
                return equation, [input_shape1, input_shape2]
            else:
                # Outer product: "i,j->ij"
                m, n = output_shape
                input_shape1 = (m,)
                input_shape2 = (n,)
                equation = "i,j->ij"
                return equation, [input_shape1, input_shape2]

        else:  # output_ndim >= 3
            # Batch operations
            if output_ndim == 3:
                # Batch matrix multiply: "bik,bkj->bij"
                b, m, n = output_shape
                k = random.randint(2, 32)
                input_shape1 = (b, m, k)
                input_shape2 = (b, k, n)
                equation = "bik,bkj->bij"
                return equation, [input_shape1, input_shape2]
            else:
                # For 4D+ outputs, use a simpler contraction pattern to avoid duplicate indices
                # Pattern: first input has some dims + contraction, second input has contraction + remaining dims
                if output_ndim == 4:
                    # 4D: "bcd,def->bcef" pattern
                    b, c, d, e = output_shape
                    k = random.randint(2, 16)
                    input_shape1 = (b, c, k)
                    input_shape2 = (k, d, e)
                    equation = "bck,kde->bcde"
                    return equation, [input_shape1, input_shape2]
                else:
                    # Fallback: use matrix multiplication pattern for higher dims
                    # Treat as batch + matrix multiply
                    batch_dims = output_shape[:-2]
                    m, n = output_shape[-2:]
                    k = random.randint(2, 16)

                    batch_size = 1
                    for dim in batch_dims:
                        batch_size *= dim

                    input_shape1 = batch_dims + (m, k)
                    input_shape2 = batch_dims + (k, n)

                    # Generate equation for batch matrix multiply
                    if len(batch_dims) == 1:
                        equation = "bik,bkj->bij"
                    else:
                        # For higher batch dims, flatten to single batch
                        equation = "bik,bkj->bij"

                    return equation, [input_shape1, input_shape2]

    def _generate_three_input_equation(self, output_shape):
        """Generate einsum equation for three input operations."""
        output_ndim = len(output_shape)

        if output_ndim == 0:
            # Scalar output - triple inner product
            vec_size = random.randint(2, 16)
            input_shape1 = (vec_size,)
            input_shape2 = (vec_size,)
            input_shape3 = (vec_size,)
            equation = "i,i,i->"
            return equation, [input_shape1, input_shape2, input_shape3]

        elif output_ndim == 1:
            # Vector output - generalized contraction
            vec_size = output_shape[0]
            k1, k2 = random.randint(2, 8), random.randint(2, 8)
            input_shape1 = (vec_size, k1)
            input_shape2 = (k1, k2)
            input_shape3 = (k2,)
            equation = "ik,kj,j->i"
            return equation, [input_shape1, input_shape2, input_shape3]

        else:
            # Higher dimensional output - batch triple product
            if output_ndim == 2:
                m, n = output_shape
                k = random.randint(2, 16)
                input_shape1 = (m, k)
                input_shape2 = (k, k)
                input_shape3 = (k, n)
                equation = "ik,kl,ln->in"
                return equation, [input_shape1, input_shape2, input_shape3]
            else:
                # Generic multi-way contraction
                b = output_shape[0] if output_ndim >= 3 else 1
                remaining = output_shape[1:] if output_ndim >= 3 else output_shape

                k1, k2 = random.randint(2, 8), random.randint(2, 8)

                if output_ndim >= 3:
                    # For 3+ dimensions, simplify to avoid duplicate indices
                    # Use a simpler pattern: batch matrix multiply extended
                    if output_ndim == 3:
                        # 3D output: batch matrix multiply
                        b, m, n = output_shape
                        input_shape1 = (b, m, k1)
                        input_shape2 = (k1, k2)
                        input_shape3 = (k2, n)
                        equation = "bik,kl,ln->bin"
                    else:
                        # 4D output: reduce to simpler batch + 2D contraction
                        b, c, d, e = output_shape
                        input_shape1 = (b, c, k1)
                        input_shape2 = (k1, k2)
                        input_shape3 = (k2, d, e)
                        equation = "bck,kl,lde->bcde"
                else:
                    # 2D case without batch
                    m, n = remaining
                    input_shape1 = (m, k1)
                    input_shape2 = (k1, k2)
                    input_shape3 = (k2, n)
                    equation = "ik,kl,ln->in"

                return equation, [input_shape1, input_shape2, input_shape3]

    def codegen(self, output_name, input_names, output_tensor):
        """Generate code for einsum operation."""
        equation = getattr(output_tensor, "_einsum_equation", None)
        if equation is None:
            raise ValueError("Einsum equation not found on output tensor")

        input_list = ", ".join(input_names)
        return f'{output_name} = torch.einsum("{equation}", {input_list})'

    def supports_variable_inputs(self) -> bool:
        """Einsum supports variable number of inputs."""
        return True
