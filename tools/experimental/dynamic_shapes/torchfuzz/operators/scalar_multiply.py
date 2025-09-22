"""Scalar multiply operator implementation."""

import random

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec


class ScalarMultiplyOperator(Operator):
    """Operator for multiplying two scalars."""

    def __init__(self):
        super().__init__("scalar_multiply")

    def can_produce(self, output_spec: Spec) -> bool:
        """Scalar multiply can only produce scalars."""
        return isinstance(output_spec, ScalarSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose scalar into input scalars for multiplication with type promotion."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError(
                "ScalarMultiplyOperator can only produce ScalarSpec outputs"
            )

        # Use shared type promotion utility
        from torchfuzz.type_promotion import get_scalar_promotion_pairs

        supported_types = get_scalar_promotion_pairs(output_spec.dtype)
        dtypes = random.choice(supported_types)

        return [ScalarSpec(dtype=dtypes[0]), ScalarSpec(dtype=dtypes[1])]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for scalar multiplication operation."""
        if len(input_names) != 2:
            raise ValueError("ScalarMultiplyOperator requires exactly two inputs")

        return f"{output_name} = {input_names[0]} * {input_names[1]}"
