# pyre-strict
"""Comparison ops + torch.where (3-arg select form only).

11 binary comparison operators (eq, ne, lt, le, gt, ge, greater,
greater_equal, less, less_equal, not_equal) plus WhereOperator.

NOTE: torch.where 1-arg form (returns int64 indices) is intentionally
NOT modeled here -- see WhereOperator docstring for rationale.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    random_broadcast_shape,
    scalar_repr,
)
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


_NUMERIC_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
)

# WhereOperator.can_produce restricts itself to this dtype set so the
# scalar forms (B and C) cannot be selected for a dtype
# _random_scalar_for_dtype does not handle.
_WHERE_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
)


def _random_scalar_for_dtype(dtype: torch.dtype) -> object:
    """Random Python scalar literal for torch.where's scalar form.

    NOTE: This helper handles ONLY the dtypes in _WHERE_SUPPORTED_DTYPES.
    Drift between the two will raise loudly at fuzz time.
    """
    if dtype == torch.bool:
        return random.choice([True, False])
    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        r = random.random()
        if r < 0.1:
            return float("-inf")
        if r < 0.15:
            return float("inf")
        return round(random.uniform(-10.0, 10.0), 4)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return random.randint(-100, 100)
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        return random.randint(0, 100)
    raise ValueError(f"Unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# Base class (excluded from introspection by the ``Base`` suffix)
# ---------------------------------------------------------------------------


class ComparisonOperatorBase(Operator):
    """Two-tensor numeric -> bool comparison.

    Both inputs share a single, randomly-chosen numeric dtype picked
    at fuzz_inputs_specs time (uses the global random module so --seed
    reproduces).
    """

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        dtype_a = random.choice(_NUMERIC_INPUT_DTYPES)
        dtype_b = random.choice(_NUMERIC_INPUT_DTYPES)
        input_dtypes = [dtype_a, dtype_b]
        spec_a = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=dtype_a,
        )
        spec_b = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=dtype_b,
        )
        specs: list[Spec] = [spec_a, spec_b]
        r = random.random()
        if r < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=input_dtypes[idx],
                )
        elif r < 0.5:
            # scalar in the `other` position only; position 0 must be a tensor
            specs[1] = ScalarSpec(dtype=input_dtypes[1])
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        )


# ---------------------------------------------------------------------------
# Comparison subclasses
# ---------------------------------------------------------------------------


class EqOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.eq")


class NeOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.ne")


class LtOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.lt")


class LeOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.le")


class GtOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.gt")


class GeOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.ge")


class GreaterOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.greater")


class GreaterEqualOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.greater_equal")


class LessOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.less")


class LessEqualOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.less_equal")


class NotEqualOperator(ComparisonOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.not_equal")


# ---------------------------------------------------------------------------
# torch.where (3-arg select form only)
# ---------------------------------------------------------------------------


class WhereOperator(Operator):
    """torch.where(cond, input, other) -- 3-arg elementwise select.

    Three forms sampled uniformly at random per call:
      A: all-tensor                  torch.where(cond, t1, t2)
      B: scalar in input position    torch.where(cond, <scalar>, t)
      C: scalar in other position    torch.where(cond, t, <scalar>)

    can_produce is restricted to _WHERE_SUPPORTED_DTYPES so Forms B and
    C never need a literal for a dtype _random_scalar_for_dtype does
    not handle.

    NOTE: The 1-arg form torch.where(cond) -> tuple[int64, ...] is
    intentionally NOT modeled. Its output shape is data-dependent on
    cond's True count, which cannot be honored under torchfuzz's
    exact-size contract (specs_compatible in tensor_fuzzer.py requires
    exact size + dtype equality). To shoehorn it in, codegen would have
    to force cond to all-True so the output length matches output_spec
    -- which is functionally equivalent to torch.arange and exercises
    no data-dependent code path. CLAUDE.md also forbids data-dependent
    ops. Skipped per design discussion.
    """

    def __init__(self) -> None:
        super().__init__("torch.where")
        self._form: str | None = None
        self._scalar_value: object | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in _WHERE_SUPPORTED_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._form = random.choice(("A", "B", "C"))
        self._scalar_value = _random_scalar_for_dtype(output_spec.dtype)
        cond_spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=torch.bool,
        )
        if random.random() < 0.3:
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                cond_spec = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=torch.bool,
                )
        body_spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        if self._form == "A":
            specs: list[Spec] = [cond_spec, body_spec, body_spec]
            if random.random() < 0.3:
                idx = random.randint(1, 2)
                bcast_size = random_broadcast_shape(tuple(output_spec.size))
                if bcast_size != tuple(output_spec.size):
                    specs[idx] = TensorSpec(
                        size=bcast_size,
                        stride=contiguous_stride(bcast_size),
                        dtype=output_spec.dtype,
                    )
            return specs
        return [cond_spec, body_spec]  # Form B or C

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        scalar_val = self._scalar_value
        if self._form == "A":
            out = (
                f"{output_name} = torch.where("
                f"{input_names[0]}, {input_names[1]}, {input_names[2]})"
            )
        elif self._form == "B":
            out = (
                f"{output_name} = torch.where("
                f"{input_names[0]}, "
                f"torch.tensor({scalar_repr(scalar_val)}, dtype={output_spec.dtype}), "
                f"{input_names[1]})"
            )
        else:  # Form C
            out = (
                f"{output_name} = torch.where("
                f"{input_names[0]}, {input_names[1]}, "
                f"torch.tensor({scalar_repr(scalar_val)}, dtype={output_spec.dtype}))"
            )
        # Defense-in-depth: clear stashed state so a stale value cannot
        # leak into the next call if upstream's lifecycle ever changes.
        self._form = None
        self._scalar_value = None
        return out
