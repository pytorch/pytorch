"""Tensor pointwise operator implementation."""

import random
from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


class PointwiseOperator(Operator):
    """Base class for element-wise pointwise operations."""

    def __init__(self, name: str, symbol: str):
        super().__init__(name)
        self.symbol = symbol

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        raise NotImplementedError("Subclasses must override torch_op_name")

    def can_produce(self, output_spec: Spec) -> bool:
        """Tensor pointwise operations can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose tensor into input tensors for pointwise operation with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
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
        if len(input_names) == 2:
            return f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        else:
            # Chain operations using symbols for readability
            expr = f" {self.symbol} ".join(input_names)
            return f"{output_name} = {expr}"


class AddOperator(PointwiseOperator):
    """Operator for element-wise addition."""

    def __init__(self, weight: float = 1.0):
        super().__init__("add", "+")
        self.weight = float(weight)

    @property
    def torch_op_name(self) -> str:
        return "torch.add"


class MulOperator(PointwiseOperator):
    """Operator for element-wise multiplication."""

    def __init__(self):
        super().__init__("mul", "*")

    @property
    def torch_op_name(self) -> str:
        return "torch.mul"


class SubOperator(PointwiseOperator):
    """Operator for element-wise subtraction."""

    def __init__(self):
        super().__init__("sub", "-")

    @property
    def torch_op_name(self) -> str:
        return "torch.sub"


class DivOperator(PointwiseOperator):
    """Operator for element-wise division."""

    def __init__(self):
        super().__init__("div", "/")

    @property
    def torch_op_name(self) -> str:
        return "torch.div"
