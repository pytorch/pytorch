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
          
        # Directly adjust to match exact element count
        if output_numel == 0:
            # For zero-sized output, create zero-sized input
            input_size = tuple(list(input_size)[:-1] + [0])
        elif len(input_size) > 0 and output_numel > 0:
            # Calculate input shape that gives exactly output_numel elements
            # Keep all dims except last, adjust last to make total = output_numel
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

        # Create input tensor spec with compatible stride
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride
        input_stride = fuzz_valid_stride(input_size)

        return [TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)]

    def codegen(self, output_name: str, input_names: list[str], output_spec: Spec) -> str:
        """Generate code for view operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ViewOperator can only produce TensorSpec outputs")

        shape_str = str(list(output_spec.size))
        return f"{output_name} = {input_names[0]}.view({shape_str})"


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
          
        # Directly adjust to match exact element count
        if output_numel == 0:
            # For zero-sized output, create zero-sized input
            input_size = tuple(list(input_size)[:-1] + [0])
        elif len(input_size) > 0 and output_numel > 0:
            # Calculate input shape that gives exactly output_numel elements
            # Keep all dims except last, adjust last to make total = output_numel
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

        # Create input tensor spec with compatible stride
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride
        input_stride = fuzz_valid_stride(input_size)

        return [TensorSpec(size=input_size, stride=input_stride, dtype=output_spec.dtype)]

    def codegen(self, output_name: str, input_names: list[str], output_spec: Spec) -> str:
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
        """Flatten can only produce 1D or 2D tensors (depending on start_dim)."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Flatten typically produces 1D or 2D tensors
        return len(output_spec.size) <= 2

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

        return [TensorSpec(size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype)]

    def codegen(self, output_name: str, input_names: list[str], output_spec: Spec) -> str:
        """Generate code for flatten operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("FlattenOperator can only produce TensorSpec outputs")

        # Choose start_dim based on output dimensions
        if len(output_spec.size) == 1:
            # Flatten all dimensions
            return f"{output_name} = torch.flatten({input_names[0]})"
        else:
            # Flatten from a random start dimension
            start_dim = random.randint(0, 1)
            return f"{output_name} = torch.flatten({input_names[0]}, start_dim={start_dim})"


class SqueezeOperator(LayoutOperatorBase):
    """Operator for torch.squeeze() operation."""

    def __init__(self):
        """Initialize SqueezeOperator."""
        super().__init__("squeeze")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.squeeze"

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for squeeze operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SqueezeOperator can only produce TensorSpec outputs")

        # Add singleton dimensions to the output shape to create input
        input_size = list(output_spec.size)

        # Randomly insert 1-sized dimensions
        num_squeeze_dims = random.randint(1, 3)
        for _ in range(num_squeeze_dims):
            pos = random.randint(0, len(input_size))
            input_size.insert(pos, 1)

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride
        input_stride = fuzz_valid_stride(tuple(input_size))

        return [TensorSpec(size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype)]

    def codegen(self, output_name: str, input_names: list[str], output_spec: Spec) -> str:
        """Generate code for squeeze operation."""
        # Randomly choose between squeeze() and squeeze(dim)
        if random.random() < 0.7:
            # Just squeeze all singleton dimensions
            return f"{output_name} = torch.squeeze({input_names[0]})"
        else:
            # Squeeze a specific dimension (we'll let PyTorch handle invalid dims)
            dim = random.randint(0, 3)
            return f"{output_name} = torch.squeeze({input_names[0]}, dim={dim})"


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

        # Remove one singleton dimension from output to create input
        input_size = list(output_spec.size)

        # Find and remove a singleton dimension
        singleton_indices = [i for i, dim in enumerate(input_size) if dim == 1]
        if singleton_indices:
            # Remove one singleton dimension
            idx_to_remove = random.choice(singleton_indices)
            input_size.pop(idx_to_remove)

        # If no singleton dimensions exist or input becomes empty, create a simpler input
        if not input_size:
            input_size = [dim for dim in output_spec.size if dim != 1]
            if not input_size:  # All dims were 1
                input_size = [1]  # Keep at least one dim

        # Create input tensor spec
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride
        input_stride = fuzz_valid_stride(tuple(input_size))

        return [TensorSpec(size=tuple(input_size), stride=input_stride, dtype=output_spec.dtype)]

    def codegen(self, output_name: str, input_names: list[str], output_spec: Spec) -> str:
        """Generate code for unsqueeze operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UnsqueezeOperator can only produce TensorSpec outputs")

        # Find where the singleton dimension should be added
        # Choose a random valid dimension position
        max_dim = len(output_spec.size)
        dim = random.randint(0, max_dim - 1)

        return f"{output_name} = torch.unsqueeze({input_names[0]}, dim={dim})"
