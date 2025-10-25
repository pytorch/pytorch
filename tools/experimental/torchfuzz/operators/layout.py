"""Tensor layout operator implementations."""

import random
from typing import Optional

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import fuzz_tensor_size, Spec, TensorSpec


class LayoutOperatorBase(Operator):
    """Base class for tensor layout operations."""

    def can_produce(self, output_spec: Spec) -> bool:
        """All layout operations can only produce tensor outputs."""
        return isinstance(output_spec, TensorSpec)


class ViewOperator(LayoutOperatorBase):
    """Operator for tensor.view() operation."""

    def __init__(self):
        """Initialize ViewOperator."""
        super().__init__("view")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.Tensor.view"

    def can_produce(self, output_spec: Spec) -> bool:
        """ViewOperator can produce tensor outputs but not scalars due to element count constraints."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Don't produce scalars since we can't guarantee input has exactly 1 element
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for view operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ViewOperator can only produce TensorSpec outputs")

        # Calculate total number of elements in output
        output_numel = 1
        for dim in output_spec.size:
            output_numel *= dim

        # Generate a compatible input shape with exactly the same number of elements
        input_size = fuzz_tensor_size()

        # Always ensure exact element count match
        if output_numel == 0:
            # For zero-sized output, create zero-sized input
            input_size = tuple(list(input_size)[:-1] + [0])
        else:
            # Calculate input shape that gives exactly output_numel elements
            # Try to use the fuzzed shape structure but adjust to match element count
            if len(input_size) > 1:
                # Keep all dims except last, adjust last to make total = output_numel
                prefix_numel = 1
                for dim in input_size[:-1]:
                    prefix_numel *= dim

                if prefix_numel > 0 and output_numel % prefix_numel == 0:
                    last_dim = output_numel // prefix_numel
                    input_size = tuple(list(input_size)[:-1] + [last_dim])
                else:
                    # Fallback: create a simple shape with exact element count
                    input_size = (output_numel,)
            else:
                # For single-dim input, just use the exact element count
                input_size = (output_numel,)

        # Create input tensor spec with contiguous stride for view compatibility
        # .view() requires compatible memory layout, so use contiguous stride
        input_stride = tuple()
        if input_size:
            # Calculate contiguous stride
            stride = [1]
            for i in range(len(input_size) - 1, 0, -1):
                stride.insert(0, stride[0] * input_size[i])
            input_stride = tuple(stride)

        return [
            TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for view operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ViewOperator can only produce TensorSpec outputs")

        shape_str = str(list(output_spec.size))
        # Ensure tensor is contiguous before view to avoid stride compatibility issues
        return f"{output_name} = {input_names[0]}.contiguous().view({shape_str})"


class ReshapeOperator(LayoutOperatorBase):
    """Operator for torch.reshape() operation."""

    def __init__(self):
        """Initialize ReshapeOperator."""
        super().__init__("reshape")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.reshape"

    def can_produce(self, output_spec: Spec) -> bool:
        """ReshapeOperator can produce tensor outputs but not scalars due to element count constraints."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Don't produce scalars since we can't guarantee input has exactly 1 element
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for reshape operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ReshapeOperator can only produce TensorSpec outputs")

        # Calculate total number of elements in output
        output_numel = 1
        for dim in output_spec.size:
            output_numel *= dim

        # Generate a compatible input shape with exactly the same number of elements
        input_size = fuzz_tensor_size()

        # Always ensure exact element count match
        if output_numel == 0:
            # For zero-sized output, create zero-sized input
            input_size = tuple(list(input_size)[:-1] + [0])
        else:
            # Calculate input shape that gives exactly output_numel elements
            # Try to use the fuzzed shape structure but adjust to match element count
            if len(input_size) > 1:
                # Keep all dims except last, adjust last to make total = output_numel
                prefix_numel = 1
                for dim in input_size[:-1]:
                    prefix_numel *= dim

                if prefix_numel > 0 and output_numel % prefix_numel == 0:
                    last_dim = output_numel // prefix_numel
                    input_size = tuple(list(input_size)[:-1] + [last_dim])
                else:
                    # Fallback: create a simple shape with exact element count
                    input_size = (output_numel,)
            else:
                # For single-dim input, just use the exact element count
                input_size = (output_numel,)

        # Create input tensor spec with compatible stride
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        input_stride = fuzz_valid_stride(input_size)

        return [
            TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for reshape operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ReshapeOperator can only produce TensorSpec outputs")

        shape_str = str(list(output_spec.size))
        return f"{output_name} = torch.reshape({input_names[0]}, {shape_str})"


class FlattenOperator(LayoutOperatorBase):
    """Operator for torch.flatten() operation."""

    def __init__(self):
        """Initialize FlattenOperator."""
        super().__init__("flatten")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.flatten"

    def can_produce(self, output_spec: Spec) -> bool:
        """Flatten can only produce 1D tensors when using torch.flatten() without start_dim."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Since we always use torch.flatten() without start_dim, we can only produce 1D tensors
        return len(output_spec.size) == 1

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for flatten operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("FlattenOperator can only produce TensorSpec outputs")

        # Calculate total number of elements in output
        output_numel = 1
        for dim in output_spec.size:
            output_numel *= dim

        # Generate a multi-dimensional input that can be flattened
        if len(output_spec.size) == 1:
            # For 1D output, generate any multi-dimensional input
            input_size = fuzz_tensor_size()
            # Ensure input has multiple dimensions
            if len(input_size) < 2:
                input_size = (2, 2)  # Default multi-dim shape
        else:
            # For 2D output, generate input with more dimensions
            input_size = fuzz_tensor_size()
            if len(input_size) < 3:
                input_size = (2, 2, 2)  # Default 3D shape

        # Adjust input size to match output element count
        input_numel = 1
        for dim in input_size:
            input_numel *= dim

        if input_numel != output_numel:
            # Handle zero-sized tensors specially
            if output_numel == 0:
                # For zero-sized output, create zero-sized input
                input_size = tuple(list(input_size)[:-1] + [0])
            elif len(input_size) > 0 and output_numel > 0:
                # Calculate input shape that gives exactly output_numel elements
                prefix_numel = 1
                for dim in input_size[:-1]:
                    prefix_numel *= dim

                if prefix_numel > 0:
                    last_dim = output_numel // prefix_numel
                    # Ensure we get exactly output_numel elements
                    if last_dim * prefix_numel == output_numel:
                        input_size = tuple(list(input_size)[:-1] + [last_dim])
                    else:
                        # Fallback: create a simple shape with exact element count
                        input_size = (output_numel,)
                else:
                    input_size = (output_numel,)

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        input_stride = fuzz_valid_stride(tuple(input_size))

        return [
            TensorSpec(
                size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for flatten operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("FlattenOperator can only produce TensorSpec outputs")

        # Always flatten all dimensions to avoid shape calculation errors
        # This ensures the output matches the expected output_spec shape
        return f"{output_name} = torch.flatten({input_names[0]})"


class SqueezeOperator(LayoutOperatorBase):
    """Operator for torch.squeeze() operation."""

    def __init__(self):
        """Initialize SqueezeOperator."""
        super().__init__("squeeze")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.squeeze"

    def can_produce(self, output_spec: Spec) -> bool:
        """SqueezeOperator can only produce tensors WITHOUT singleton dimensions."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Don't produce outputs with singleton dimensions since squeeze() removes ALL of them
        return 1 not in output_spec.size

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for squeeze operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SqueezeOperator can only produce TensorSpec outputs")

        # Add exactly one singleton dimension to the output shape to create input
        input_size = list(output_spec.size)
        # Insert exactly one singleton dimension at a random position
        pos = random.randint(0, len(input_size))
        input_size.insert(pos, 1)

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        input_stride = fuzz_valid_stride(tuple(input_size))

        return [
            TensorSpec(
                size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for squeeze operation."""
        # Always use squeeze() without dim specification to be safe
        # Since we control input generation to add exactly one singleton dimension,
        # and we preserve existing singleton dimensions in the output,
        # this should work correctly
        return f"{output_name} = torch.squeeze({input_names[0]})"


class UnsqueezeOperator(LayoutOperatorBase):
    """Operator for torch.unsqueeze() operation."""

    def __init__(self):
        """Initialize UnsqueezeOperator."""
        super().__init__("unsqueeze")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.unsqueeze"

    def can_produce(self, output_spec: Spec) -> bool:
        """Unsqueeze produces tensors with at least one singleton dimension."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Check if there's at least one singleton dimension
        return 1 in output_spec.size

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for unsqueeze operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UnsqueezeOperator can only produce TensorSpec outputs")

        # For unsqueeze: output = input.shape[:dim] + (1,) + input.shape[dim:]
        # So to get input from output, we need to remove exactly one singleton dimension

        # Find a singleton dimension to remove (prefer last one for consistency)
        input_size = list(output_spec.size)
        singleton_idx = None

        for i in range(len(input_size) - 1, -1, -1):
            if input_size[i] == 1:
                singleton_idx = i
                break

        if singleton_idx is not None:
            # Remove the singleton dimension to create input shape
            input_size.pop(singleton_idx)
        else:
            # This shouldn't happen given our can_produce constraint
            raise ValueError(
                "UnsqueezeOperator requires output to have at least one singleton dimension"
            )

        # Handle empty input (scalar case)
        if not input_size:
            input_size = tuple()  # Scalar tensor
        else:
            input_size = tuple(input_size)

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        if input_size:
            input_stride = fuzz_valid_stride(input_size)
        else:
            input_stride = tuple()  # Scalar has empty stride

        return [
            TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for unsqueeze operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UnsqueezeOperator can only produce TensorSpec outputs")

        # Find the last singleton dimension position (matching fuzz_inputs_specs logic)
        # This should be the same singleton dimension that we removed in fuzz_inputs_specs
        last_singleton_idx = None
        for i in range(len(output_spec.size) - 1, -1, -1):
            if output_spec.size[i] == 1:
                last_singleton_idx = i
                break

        if last_singleton_idx is not None:
            dim = last_singleton_idx
        else:
            # Fallback: add at the end (shouldn't happen given our can_produce constraint)
            dim = len(output_spec.size) - 1

        return f"{output_name} = torch.unsqueeze({input_names[0]}, dim={dim})"


class SplitOperator(LayoutOperatorBase):
    """Operator for torch.split() operation."""

    def __init__(self):
        """Initialize SplitOperator."""
        super().__init__("split")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.split"

    def can_produce(self, output_spec: Spec) -> bool:
        """Split can produce any tensor output."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Split can produce any tensor with at least one dimension
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for split operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SplitOperator can only produce TensorSpec outputs")

        # torch.split() splits a tensor along a dimension
        # We'll use split_size_or_sections as an integer (split_size)
        # The output will be one of the chunks from the split

        # Choose a random dimension to split along
        if len(output_spec.size) == 0:
            raise ValueError("Cannot split a scalar tensor")

        split_dim = random.randint(0, len(output_spec.size) - 1)

        # Choose a random number of chunks (at least 2)
        num_chunks = random.randint(2, 4)

        # Calculate input size: input will have split_dim with size = output_size * num_chunks
        # (or slightly larger to account for uneven splits)
        input_size = list(output_spec.size)
        input_size[split_dim] = output_spec.size[split_dim] * num_chunks

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        input_stride = fuzz_valid_stride(tuple(input_size))

        return [
            TensorSpec(
                size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for split operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SplitOperator can only produce TensorSpec outputs")

        # Choose a random dimension to split along
        split_dim = random.randint(0, len(output_spec.size) - 1)

        # Use output size along split_dim as the split_size
        split_size = output_spec.size[split_dim]

        # Generate the split and select the first chunk
        return f"{output_name} = torch.split({input_names[0]}, {split_size}, dim={split_dim})[0]"


class ExpandOperator(LayoutOperatorBase):
    """Operator for torch.expand() operation."""

    def __init__(self):
        """Initialize ExpandOperator."""
        super().__init__("expand")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.expand"

    def can_produce(self, output_spec: Spec) -> bool:
        """Expand can produce any tensor output."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Expand can produce any tensor with at least one dimension
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for expand operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ExpandOperator can only produce TensorSpec outputs")

        # torch.expand() broadcasts a tensor to a new shape
        # For expand to work, each dimension of the input must either:
        # 1. Match the corresponding output dimension
        # 2. Be 1 (to be broadcasted)
        # 3. Not exist (input can have fewer dimensions than output)

        # Generate input size with same or fewer dimensions
        output_size = output_spec.size
        input_ndim = random.randint(1, len(output_size))

        # Create input size by choosing dimensions to broadcast
        input_size = []
        for i in range(input_ndim):
            output_dim_idx = len(output_size) - input_ndim + i
            output_dim = output_size[output_dim_idx]

            # Randomly choose to either match the output dimension or use 1 for broadcasting
            # Use 1 with higher probability to test broadcasting behavior
            if random.random() < 0.6 and output_dim > 1:
                input_size.append(1)
            else:
                input_size.append(output_dim)

        input_size = tuple(input_size)

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        input_stride = fuzz_valid_stride(input_size)

        return [
            TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for expand operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ExpandOperator can only produce TensorSpec outputs")

        shape_str = str(list(output_spec.size))
        return f"{output_name} = {input_names[0]}.expand({shape_str})"
