"""Matrix multiplication operator implementations."""

import random

import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# Type promotion imports removed since we now use explicit casting in codegen


class MatrixMultiplyOperator(Operator):
    """Base class for matrix multiplication operations."""

    def __init__(self, name: str):
        super().__init__(name)

    def can_produce(self, output_spec: Spec) -> bool:
        """Matrix multiply operations can produce float/complex tensors of dimension >= 2."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have at least 2 dimensions for matrix multiplication
        if len(output_spec.size) < 2:
            return False

        # Matrix multiply doesn't work with bool or integer types for gradients
        if output_spec.dtype in [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return False

        return True

    def _get_compatible_dtype(self, output_dtype):
        """Get a compatible dtype for matrix multiplication."""
        return [output_dtype, output_dtype]


class MMOperator(MatrixMultiplyOperator):
    """Operator for matrix multiplication (torch.mm)."""

    def __init__(self):
        super().__init__("mm")
        self.weight = 5.0

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.mm"

    def can_produce(self, output_spec: Spec) -> bool:
        """MM requires exactly 2D tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have exactly 2 dimensions for torch.mm
        if len(output_spec.size) != 2:
            return False

        # Matrix multiply doesn't work with bool or integer types for gradients
        if output_spec.dtype in [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return False

        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for matrix multiplication."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("MMOperator can only produce TensorSpec outputs")

        if len(output_spec.size) != 2:
            raise ValueError("torch.mm requires 2D tensors")

        m, n = output_spec.size
        # Choose a random inner dimension k
        k = random.randint(1, 16)

        dtypes = self._get_compatible_dtype(output_spec.dtype)

        # First tensor: [m, k]
        input1_spec = TensorSpec(
            size=(m, k),
            stride=(k, 1),  # Contiguous stride
            dtype=dtypes[0],
        )

        # Second tensor: [k, n]
        input2_spec = TensorSpec(
            size=(k, n),
            stride=(n, 1),  # Contiguous stride
            dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
        )

        return [input1_spec, input2_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for matrix multiplication."""
        if len(input_names) != 2:
            raise ValueError("torch.mm requires exactly 2 inputs")

        # Get target dtype
        if isinstance(output_spec, TensorSpec):
            target_dtype_str = f"torch.{output_spec.dtype}".replace(
                "torch.torch.", "torch."
            )
            # Cast inputs to ensure compatible types
            return (
                f"{output_name} = torch.mm("
                f"{input_names[0]}.to({target_dtype_str}), "
                f"{input_names[1]}.to({target_dtype_str}))"
            )
        else:
            return f"{output_name} = torch.mm({input_names[0]}, {input_names[1]})"


class AddmmOperator(MatrixMultiplyOperator):
    """Operator for additive matrix multiplication (torch.addmm)."""

    def __init__(self):
        super().__init__("addmm")
        self.weight = 5.0

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.addmm"

    def can_produce(self, output_spec: Spec) -> bool:
        """Addmm requires exactly 2D tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have exactly 2 dimensions for torch.addmm
        if len(output_spec.size) != 2:
            return False

        # Matrix multiply doesn't work with bool or integer types for gradients
        if output_spec.dtype in [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return False

        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for additive matrix multiplication."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("AddmmOperator can only produce TensorSpec outputs")

        if len(output_spec.size) != 2:
            raise ValueError("torch.addmm requires 2D output tensor")

        m, n = output_spec.size
        # Choose a random inner dimension k
        k = random.randint(1, 16)

        dtypes = self._get_compatible_dtype(output_spec.dtype)

        # Bias tensor: [m, n] (same shape as output)
        bias_spec = TensorSpec(
            size=(m, n),
            stride=(n, 1),  # Contiguous stride
            dtype=dtypes[0],
        )

        # First matrix: [m, k]
        input1_spec = TensorSpec(
            size=(m, k),
            stride=(k, 1),  # Contiguous stride
            dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
        )

        # Second matrix: [k, n]
        input2_spec = TensorSpec(
            size=(k, n),
            stride=(n, 1),  # Contiguous stride
            dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
        )

        return [bias_spec, input1_spec, input2_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for additive matrix multiplication."""
        if len(input_names) != 3:
            raise ValueError("torch.addmm requires exactly 3 inputs")

        # Get target dtype
        if isinstance(output_spec, TensorSpec):
            target_dtype_str = f"torch.{output_spec.dtype}".replace(
                "torch.torch.", "torch."
            )
            # Cast inputs to ensure compatible types
            return (
                f"{output_name} = torch.addmm("
                f"{input_names[0]}.to({target_dtype_str}), "
                f"{input_names[1]}.to({target_dtype_str}), "
                f"{input_names[2]}.to({target_dtype_str}))"
            )
        else:
            return f"{output_name} = torch.addmm({input_names[0]}, {input_names[1]}, {input_names[2]})"


class BmmOperator(MatrixMultiplyOperator):
    """Operator for batch matrix multiplication (torch.bmm)."""

    def __init__(self):
        super().__init__("bmm")
        self.weight = 5.0

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.bmm"

    def can_produce(self, output_spec: Spec) -> bool:
        """Batch matrix multiply requires 3D tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have exactly 3 dimensions for batch matrix multiplication
        if len(output_spec.size) != 3:
            return False

        # Matrix multiply doesn't work with bool or integer types for gradients
        if output_spec.dtype in [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return False

        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for batch matrix multiplication."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("BmmOperator can only produce TensorSpec outputs")

        if len(output_spec.size) != 3:
            raise ValueError("torch.bmm requires 3D tensors")

        b, m, n = output_spec.size
        # Choose a random inner dimension k
        k = random.randint(1, 16)

        dtypes = self._get_compatible_dtype(output_spec.dtype)

        # First tensor: [b, m, k]
        input1_spec = TensorSpec(
            size=(b, m, k),
            stride=(m * k, k, 1),  # Contiguous stride
            dtype=dtypes[0],
        )

        # Second tensor: [b, k, n]
        input2_spec = TensorSpec(
            size=(b, k, n),
            stride=(k * n, n, 1),  # Contiguous stride
            dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
        )

        return [input1_spec, input2_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for batch matrix multiplication."""
        if len(input_names) != 2:
            raise ValueError("torch.bmm requires exactly 2 inputs")

        # Get target dtype
        if isinstance(output_spec, TensorSpec):
            target_dtype_str = f"torch.{output_spec.dtype}".replace(
                "torch.torch.", "torch."
            )
            # Cast inputs to ensure compatible types
            return (
                f"{output_name} = torch.bmm("
                f"{input_names[0]}.to({target_dtype_str}), "
                f"{input_names[1]}.to({target_dtype_str}))"
            )
        else:
            return f"{output_name} = torch.bmm({input_names[0]}, {input_names[1]})"


class MatmulOperator(MatrixMultiplyOperator):
    """Operator for general matrix multiplication (torch.matmul)."""

    def __init__(self):
        super().__init__("matmul")
        self.weight = 500.0

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.matmul"

    def can_produce(self, output_spec: Spec) -> bool:
        """Matmul can handle various tensor dimensions >= 1."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have at least 1 dimension
        if len(output_spec.size) < 1:
            return False

        # Matrix multiply doesn't work with bool or integer types for gradients
        if output_spec.dtype in [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return False

        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for general matrix multiplication."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("MatmulOperator can only produce TensorSpec outputs")

        output_size = output_spec.size
        output_dims = len(output_size)

        dtypes = self._get_compatible_dtype(output_spec.dtype)

        if output_dims == 1:
            # Matrix-vector multiplication: (n,) = (k,) @ (k, n) or (n,) = (n, k) @ (k,)
            n = output_size[0]
            k = random.randint(1, 16)

            # Randomly choose between two valid patterns
            if random.choice([True, False]):
                # Pattern 1: (n,) = (k,) @ (k, n)
                input1_spec = TensorSpec(size=(k,), stride=(1,), dtype=dtypes[0])
                input2_spec = TensorSpec(
                    size=(k, n),
                    stride=(n, 1),
                    dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
                )
            else:
                # Pattern 2: (n,) = (n, k) @ (k,)
                input1_spec = TensorSpec(size=(n, k), stride=(k, 1), dtype=dtypes[0])
                input2_spec = TensorSpec(
                    size=(k,),
                    stride=(1,),
                    dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
                )

        elif output_dims == 2:
            # Matrix multiplication: (m, n) = (m, k) @ (k, n)
            m, n = output_size
            k = random.randint(1, 16)

            input1_spec = TensorSpec(size=(m, k), stride=(k, 1), dtype=dtypes[0])
            input2_spec = TensorSpec(
                size=(k, n),
                stride=(n, 1),
                dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
            )

        else:
            # Batched matrix multiplication: (..., m, n) = (..., m, k) @ (..., k, n)
            *batch_dims, m, n = output_size
            k = random.randint(1, 16)

            # Calculate strides for contiguous tensors
            input1_size = tuple(batch_dims + [m, k])
            input2_size = tuple(batch_dims + [k, n])

            # Contiguous strides
            input1_stride = [1]
            for i in reversed(range(len(input1_size) - 1)):
                input1_stride.append(input1_stride[-1] * input1_size[i + 1])
            input1_stride = tuple(reversed(input1_stride))

            input2_stride = [1]
            for i in reversed(range(len(input2_size) - 1)):
                input2_stride.append(input2_stride[-1] * input2_size[i + 1])
            input2_stride = tuple(reversed(input2_stride))

            input1_spec = TensorSpec(
                size=input1_size, stride=input1_stride, dtype=dtypes[0]
            )
            input2_spec = TensorSpec(
                size=input2_size,
                stride=input2_stride,
                dtype=dtypes[1] if len(dtypes) > 1 else dtypes[0],
            )

        return [input1_spec, input2_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for general matrix multiplication."""
        if len(input_names) != 2:
            raise ValueError("torch.matmul requires exactly 2 inputs")

        # Get target dtype
        if isinstance(output_spec, TensorSpec):
            target_dtype_str = f"torch.{output_spec.dtype}".replace(
                "torch.torch.", "torch."
            )
            # Cast inputs to ensure compatible types
            return (
                f"{output_name} = torch.matmul("
                f"{input_names[0]}.to({target_dtype_str}), "
                f"{input_names[1]}.to({target_dtype_str}))"
            )
        else:
            return f"{output_name} = torch.matmul({input_names[0]}, {input_names[1]})"
