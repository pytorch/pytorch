# pyre-strict
"""Bitwise ops: and, or, xor, not, left_shift, right_shift.

and/or/xor/not accept integer + bool dtypes (output dtype = input dtype).
left_shift / right_shift accept integer dtypes only (no bool).

Shift-count values are NOT constrained by default -- out-of-range shifts
at runtime are tolerated as 'fuzzer-fuzzy' behavior. An opt-in clamp is
available via _CLAMP_SHIFT_COUNTS to filter UB-flavored noise during
triage.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import contiguous_stride, random_broadcast_shape
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


_BITWISE_DTYPES: tuple[torch.dtype, ...] = (
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

_SHIFT_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)

# Opt-in noise filter: when True, shift codegen wraps the shift-count
# input in torch.clamp(0, bit_width-1) so out-of-range UB does not flood
# triage. Default False preserves the fuzzer-fuzzy behavior intended by
# the design discussion.
_CLAMP_SHIFT_COUNTS: bool = False


_BIT_WIDTHS: dict[torch.dtype, int] = {
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.uint8: 8,
    torch.uint16: 16,
    torch.uint32: 32,
    torch.uint64: 64,
}


# ---------------------------------------------------------------------------
# Base classes (excluded from introspection by the ``Base`` suffix)
# ---------------------------------------------------------------------------


class BitwiseBinaryOperatorBase(Operator):
    """Two-tensor bitwise. Inputs match output dtype/shape/stride.

    Allowed dtypes: int8/16/32/64, uint8/16/32/64, bool.
    """

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _BITWISE_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        r = random.random()
        if r < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        elif r < 0.5:
            # scalar operand
            idx = random.randint(0, 1)
            specs[idx] = ScalarSpec(dtype=output_spec.dtype)
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


class BitwiseShiftOperatorBase(Operator):
    """Two-tensor bitwise shift. Inputs match output dtype/shape/stride.

    Allowed dtypes: int8/16/32/64, uint8/16/32/64 (no bool).

    The second input (shift count) is a same-dtype, same-shape tensor.
    By default values are NOT constrained -- out-of-range shifts at
    runtime are accepted as fuzzer-fuzzy behavior. When
    _CLAMP_SHIFT_COUNTS is True (a one-line module toggle), the shift
    count is wrapped in torch.clamp(0, bit_width - 1) at codegen time
    to filter UB-flavored noise during triage.
    """

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _SHIFT_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        r = random.random()
        if r < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        elif r < 0.5:
            # scalar shift count
            idx = random.randint(0, 1)
            specs[idx] = ScalarSpec(dtype=output_spec.dtype)
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        if _CLAMP_SHIFT_COUNTS:
            bw = _BIT_WIDTHS[output_spec.dtype]
            shift_expr = f"torch.clamp({input_names[1]}, 0, {bw - 1})"
        else:
            shift_expr = input_names[1]
        return f"{output_name} = {self.torch_op_name}({input_names[0]}, {shift_expr})"


# ---------------------------------------------------------------------------
# Bitwise binary subclasses
# ---------------------------------------------------------------------------


class BitwiseAndOperator(BitwiseBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.bitwise_and")


class BitwiseOrOperator(BitwiseBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.bitwise_or")


class BitwiseXorOperator(BitwiseBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.bitwise_xor")


# ---------------------------------------------------------------------------
# Bitwise shift subclasses
# ---------------------------------------------------------------------------


class BitwiseLeftShiftOperator(BitwiseShiftOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.bitwise_left_shift")


class BitwiseRightShiftOperator(BitwiseShiftOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.bitwise_right_shift")


# ---------------------------------------------------------------------------
# Bitwise unary
# ---------------------------------------------------------------------------


class BitwiseNotOperator(Operator):
    """Single-tensor bitwise NOT. Same dtype set as bitwise binary."""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def __init__(self) -> None:
        super().__init__("torch.bitwise_not")

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _BITWISE_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        return f"{output_name} = torch.bitwise_not({input_names[0]})"
