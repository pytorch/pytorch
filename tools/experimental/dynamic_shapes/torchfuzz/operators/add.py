"""Add operator implementation."""

import random

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


class AddOperator(Operator):
    """Operator for element-wise addition."""

    def __init__(self):
        super().__init__("torch.ops.aten.add")

    def can_produce(self, output_spec: Spec) -> bool:
        """Add can produce tensors but not scalars."""
        return isinstance(output_spec, TensorSpec)

    def supports_variable_inputs(self) -> bool:
        """Add operator supports variable number of inputs."""
        return True

    def decompose(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose tensor into input tensors for addition with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("AddOperator can only produce TensorSpec outputs")

        # Use shared type promotion table
        promotion_table = get_promotion_table_for_strings()

        # If num_inputs > 2, promote left-to-right (e.g. (((a + b) + c) + d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype_str = get_dtype_name(output_spec.dtype)
        supported_types = promotion_table.get(dtype_str, [(dtype_str, dtype_str)])

        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype_str)
        else:
            dtypes = [dtype_str] * num_inputs

        # Convert dtype strings back to torch dtypes
        dtype_map = get_dtype_map()

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype_map.get(dt, output_spec.dtype),
            )
            for dt in dtypes
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for addition operation."""
        if len(input_names) == 2:
            return f"{output_name} = torch.ops.aten.add({input_names[0]}, {input_names[1]})"
        else:
            # Sum all input tensors
            expr = " + ".join(input_names)
            return f"{output_name} = {expr}"
