"""Tensor pointwise operator implementation."""

import random

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


class TensorPointwiseOperator(Operator):
    """Operator for element-wise pointwise operations (add, mul, sub, div)."""

    def __init__(self):
        super().__init__("tensor_pointwise")
        self.operations = {
            "add": {
                "torch_op": "torch.ops.aten.add",
                "symbol": "+",
            },
            "mul": {
                "torch_op": "torch.ops.aten.mul",
                "symbol": "*",
            },
            "sub": {
                "torch_op": "torch.ops.aten.sub",
                "symbol": "-",
            },
            "div": {
                "torch_op": "torch.ops.aten.div",
                "symbol": "/",
            },
        }

    def can_produce(self, output_spec: Spec) -> bool:
        """Tensor pointwise operations can produce tensors but not scalars."""
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose tensor into input tensors for pointwise operation with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "TensorPointwiseOperator can only produce TensorSpec outputs"
            )

        # Use shared type promotion table
        promotion_table = get_promotion_table_for_strings()

        # If num_inputs > 2, promote left-to-right (e.g. (((a op b) op c) op d))
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
        """Generate code for pointwise operation."""
        # Randomly choose an operation
        op_name = random.choice(list(self.operations.keys()))
        op_info = self.operations[op_name]

        if len(input_names) == 2:
            return f"{output_name} = {op_info['torch_op']}({input_names[0]}, {input_names[1]})"
        else:
            # Chain operations using symbols for readability
            expr = f" {op_info['symbol']} ".join(input_names)
            return f"{output_name} = {expr}"
