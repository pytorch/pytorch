"""Collective communication operator implementations for distributed fuzzing."""

import random

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class CollectiveOperator(Operator):
    """Base class for collective communication operations."""

    def __init__(self, name: str, weight: float = 10.0):
        super().__init__(name, weight)

    def can_produce(self, output_spec: Spec) -> bool:
        """Collectives work on float tensors with at least 1 dimension."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Must have at least 1 dimension
        if len(output_spec.size) < 1:
            return False

        # Collectives typically work best with float types
        if output_spec.dtype in [torch.bool]:
            return False

        # Need non-empty first dimension for scatter/gather
        if output_spec.size[0] == 0:
            return False

        return True


class AllGatherOperator(CollectiveOperator):
    """Operator for all_gather_into_tensor collective.

    all_gather concatenates tensors from all ranks along dim 0.
    Output shape: (input_shape[0] * world_size, *input_shape[1:])

    For fuzzing, we work backwards: given output shape, compute input shape
    by dividing first dim by world_size.
    """

    def __init__(self):
        super().__init__("all_gather")
        self.weight = 200.0  # High weight to ensure collectives are selected

    @property
    def torch_op_name(self) -> str | None:
        return "torch.ops._c10d_functional.all_gather_into_tensor"

    def can_produce(self, output_spec: Spec) -> bool:
        if not super().can_produce(output_spec):
            return False

        if not isinstance(output_spec, TensorSpec):
            return False

        # First dim must be divisible by world_size=8 (matching template)
        first_dim = output_spec.size[0]
        return first_dim >= 8 and first_dim % 8 == 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for all_gather.

        Input is the local shard, output is gathered result.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("AllGatherOperator can only produce TensorSpec outputs")

        first_dim = output_spec.size[0]

        # Use world_size=8 (matching the template's group_size)
        world_size = 8

        # Input shape has first dim divided by world_size
        input_size = (first_dim // world_size,) + output_spec.size[1:]

        # Compute contiguous stride
        stride = []
        s = 1
        for dim in reversed(input_size):
            stride.append(s)
            s *= dim if dim > 0 else 1
        stride = tuple(reversed(stride))

        input_spec = TensorSpec(
            size=input_size,
            stride=stride,
            dtype=output_spec.dtype,
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for all_gather_into_tensor."""
        if len(input_names) != 1:
            raise ValueError("all_gather requires exactly 1 input")

        # We need to infer world_size from input/output shapes
        # The codegen needs access to the shapes which we encode in the spec
        if isinstance(output_spec, TensorSpec):
            # Generate code that:
            # 1. Calls all_gather_into_tensor with group info
            # 2. Waits for completion
            return f"""{output_name}_async = torch.ops._c10d_functional.all_gather_into_tensor({input_names[0]}.contiguous(), group_size, group_name)
{output_name} = torch.ops._c10d_functional.wait_tensor({output_name}_async)"""
        else:
            return f"{output_name} = {input_names[0]}"


class ReduceScatterOperator(CollectiveOperator):
    """Operator for reduce_scatter_tensor collective.

    reduce_scatter reduces tensors across ranks then scatters the result.
    Output shape: (input_shape[0] // world_size, *input_shape[1:])

    For fuzzing, we work backwards: given output shape, compute input shape
    by multiplying first dim by world_size.
    """

    def __init__(self):
        super().__init__("reduce_scatter")
        self.weight = 200.0  # High weight to ensure collectives are selected

    @property
    def torch_op_name(self) -> str | None:
        return "torch.ops._c10d_functional.reduce_scatter_tensor"

    def can_produce(self, output_spec: Spec) -> bool:
        if not super().can_produce(output_spec):
            return False

        if not isinstance(output_spec, TensorSpec):
            return False

        # Output can be any size, input will be scaled up
        # But we need reasonable bounds
        return output_spec.size[0] >= 1

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for reduce_scatter.

        Input is the full tensor, output is the scattered shard.
        Input first dim must be divisible by world_size.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ReduceScatterOperator can only produce TensorSpec outputs")

        # Use world_size=8 (matching the template's group_size)
        # This ensures consistency with the process group setup
        world_size = 8

        # Input shape has first dim multiplied by world_size
        input_size = (output_spec.size[0] * world_size,) + output_spec.size[1:]

        # Compute contiguous stride
        stride = []
        s = 1
        for dim in reversed(input_size):
            stride.append(s)
            s *= dim if dim > 0 else 1
        stride = tuple(reversed(stride))

        input_spec = TensorSpec(
            size=input_size,
            stride=stride,
            dtype=output_spec.dtype,
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for reduce_scatter_tensor."""
        if len(input_names) != 1:
            raise ValueError("reduce_scatter requires exactly 1 input")

        return f"""{output_name}_async = torch.ops._c10d_functional.reduce_scatter_tensor({input_names[0]}.contiguous(), "sum", group_size, group_name)
{output_name} = torch.ops._c10d_functional.wait_tensor({output_name}_async)"""


class AllReduceOperator(CollectiveOperator):
    """Operator for all_reduce collective.

    all_reduce reduces tensors across all ranks (sum by default).
    Output shape is same as input shape.
    """

    def __init__(self):
        super().__init__("all_reduce")
        self.weight = 200.0  # High weight to ensure collectives are selected

    @property
    def torch_op_name(self) -> str | None:
        return "torch.ops._c10d_functional.all_reduce"

    def can_produce(self, output_spec: Spec) -> bool:
        if not super().can_produce(output_spec):
            return False

        if not isinstance(output_spec, TensorSpec):
            return False

        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for all_reduce.

        Input and output have the same shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("AllReduceOperator can only produce TensorSpec outputs")

        # Input is same shape as output
        input_spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for all_reduce."""
        if len(input_names) != 1:
            raise ValueError("all_reduce requires exactly 1 input")

        return f"""{output_name}_async = torch.ops._c10d_functional.all_reduce({input_names[0]}.contiguous(), "sum", group_name)
{output_name} = torch.ops._c10d_functional.wait_tensor({output_name}_async)"""
