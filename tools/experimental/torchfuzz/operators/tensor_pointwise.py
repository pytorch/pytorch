"""Tensor pointwise operator implementation."""

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def _random_broadcast_shape(output_size: tuple[int, ...]) -> tuple[int, ...]:
    if not output_size:
        return ()
    result = list(output_size)
    changed = False
    for i in range(len(result)):
        if random.random() < 0.3:
            result[i] = 1
            changed = True
    if not changed:
        return output_size
    while len(result) > 1 and result[0] == 1 and random.random() < 0.5:
        result.pop(0)
    return tuple(result)


class PointwiseOperatorBase(Operator):
    """Base class for element-wise pointwise operations."""

    _scalar_input_positions: tuple[int, ...] = (0, 1)

    def __init__(self, name: str, symbol: str = ""):
        super().__init__(name)
        self.symbol = symbol

    @property
    def torch_op_name(self) -> str:
        return self.name

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

        input_specs: list[Spec] = [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype_map.get(dt, output_spec.dtype),
            )
            for dt in dtypes
        ]

        if num_inputs >= 2:
            r = random.random()
            if self._scalar_input_positions and r < 0.3:
                idx = random.choice(self._scalar_input_positions)
                input_specs[idx] = ScalarSpec(dtype=input_specs[idx].dtype)
            elif r < 0.5:
                idx = random.randint(0, 1)
                bcast_size = _random_broadcast_shape(tuple(output_spec.size))
                if bcast_size != tuple(output_spec.size):
                    input_specs[idx] = TensorSpec(
                        size=bcast_size,
                        stride=_contiguous_stride(bcast_size),
                        dtype=input_specs[idx].dtype,
                    )

        return input_specs

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


class AddOperator(PointwiseOperatorBase):
    """Operator for element-wise addition."""

    def __init__(self, weight: float = 1.0):
        super().__init__("add", "+")
        self.weight = float(weight)

    @property
    def torch_op_name(self) -> str:
        return "torch.add"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if len(input_names) == 2 and random.random() < 0.3:
            alpha = round(random.uniform(-5.0, 5.0), 4)
            if isinstance(output_spec, TensorSpec) and output_spec.dtype not in (
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ):
                alpha = int(alpha)
            return (
                f"{output_name} = torch.add("
                f"{input_names[0]}, {input_names[1]}, alpha={alpha!r})"
            )
        return super().codegen(output_name, input_names, output_spec)


class MulOperator(PointwiseOperatorBase):
    """Operator for element-wise multiplication."""

    def __init__(self):
        super().__init__("mul", "*")

    @property
    def torch_op_name(self) -> str:
        return "torch.mul"


class SubOperator(PointwiseOperatorBase):
    """Operator for element-wise subtraction."""

    def __init__(self):
        super().__init__("sub", "-")

    @property
    def torch_op_name(self) -> str:
        return "torch.sub"


class DivOperator(PointwiseOperatorBase):
    """Operator for element-wise division."""

    def __init__(self):
        super().__init__("div", "/")

    @property
    def torch_op_name(self) -> str:
        return "torch.div"


class ClampOperator(Operator):
    """Operator for torch.clamp (element-wise clamping)."""

    def __init__(self):
        super().__init__("clamp")

    @property
    def torch_op_name(self) -> str:
        return "torch.clamp"

    def can_produce(self, output_spec: Spec) -> bool:
        """Clamp can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for clamp operation.

        Clamp takes:
        - input tensor (same shape and dtype as output)
        - optional min value (scalar or None)
        - optional max value (scalar or None)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ClampOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for clamp operation."""
        if len(input_names) != 1:
            raise ValueError("Clamp requires exactly 1 input tensor")

        input_name = input_names[0]

        # Generate random min and max values for clamping
        # We'll randomly decide whether to use min, max, or both
        use_min = random.random() > 0.33
        use_max = random.random() > 0.33

        if not use_min and not use_max:
            use_min = True

        # For integer-dtype tensors, emit integer bounds so the fuzzer doesn't
        # generate float-bound + int-tensor mismatches between CPU and MTIA.
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        if output_spec.dtype in FLOAT_DTYPES:
            min_literal, max_literal = "-1.0", "1.0"
        else:
            min_literal, max_literal = "-1", "1"

        args = [input_name]
        if use_min:
            args.append(f"min={min_literal}")
        else:
            args.append("min=None")

        if use_max:
            args.append(f"max={max_literal}")
        else:
            args.append("max=None")

        return f"{output_name} = torch.clamp({', '.join(args)})"


class CumsumOperator(Operator):
    """Operator for torch.cumsum (cumulative sum along a dimension)."""

    def __init__(self):
        super().__init__("cumsum")

    @property
    def torch_op_name(self) -> str:
        return "torch.cumsum"

    def can_produce(self, output_spec: Spec) -> bool:
        """Cumsum can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        # Cumsum needs at least 1 dimension
        if isinstance(output_spec, TensorSpec) and len(output_spec.size) == 0:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for cumsum operation.

        Cumsum takes an input tensor with same shape and dtype as output.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("CumsumOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for cumsum operation."""
        if len(input_names) != 1:
            raise ValueError("Cumsum requires exactly 1 input tensor")

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("Output spec must be a TensorSpec")

        input_name = input_names[0]

        # Choose a random valid dimension
        num_dims = len(output_spec.size)
        if num_dims == 0:
            raise ValueError("Cumsum requires tensor with at least 1 dimension")

        # Pick a random dimension index
        dim = random.randint(0, num_dims - 1)

        return f"{output_name} = torch.cumsum({input_name}, dim={dim})"
